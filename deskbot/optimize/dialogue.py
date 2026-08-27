"""自动化对话考核（文字层）。

驱动 DeskBot._handle 的分支逻辑（退出/位置/记忆/反馈/正常问答），
用预期行为断言判定每个对话场景是否通过。不涉及 TTS/音频。

用法：python -m deskbot.main --dialogue
报告追加到 data/dialogue_report.jsonl（格式对齐 eval_report）。
"""
from __future__ import annotations

import asyncio
import io
import re
from contextlib import redirect_stdout
from typing import Dict, List, Optional

from ..main import DeskBot, _is_noise
from ..utils.config import Config
from ..utils.logging_setup import get_logger
from ..utils.state import append_jsonl, now_ts
from .feedback import FeedbackCollector

log = get_logger("optimize.dialogue")

_DATE_RE = re.compile(r"今天是\d+年\d+月\d+日")


class FakeLlm:
    """记录每次 ask 调用并返回罐头答案；维护多轮 history。

    成员与真实 LlmClient.ask 对齐（rag_context/learned_rules/vision_desc）。
    """

    def __init__(self) -> None:
        self.calls: List[dict] = []
        self.history: List[dict] = []
        self.answers: Dict[str, str] = {}   # 精确输入 → 答案；无则用 default_answer
        self.default_answer = "（测试回答）"
        self.complete_calls: List[tuple] = []

    def reset(self) -> None:
        self.calls.clear()
        self.history.clear()
        self.complete_calls.clear()

    # ---- 与真实 LlmClient 对齐的接口 ----
    async def ask(self, user_text, rag_context="", learned_rules="", vision_desc="", stream=False) -> str:
        self.calls.append({"user": user_text, "rag": rag_context,
                           "rules": learned_rules, "vision": vision_desc})
        answer = self.answers.get(user_text, self.default_answer)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    async def ask_stream(self, user_text, rag_context="", learned_rules="", vision_desc=""):
        self.calls.append({"user": user_text, "rag": rag_context,
                           "rules": learned_rules, "vision": vision_desc})
        answer = self.answers.get(user_text, self.default_answer)
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        for sent in re.split(r"([。！？])", answer):
            if sent:
                yield sent

    async def complete(self, system, user, max_tokens=None) -> str:
        self.complete_calls.append((system, user, max_tokens))
        return "[]"

    @property
    def provider(self) -> str:
        return "fake"


def _make_bot(cfg: Config) -> DeskBot:
    """构造 voice=False 的 DeskBot，并替换为可离线/确定性运行的组件。"""
    bot = DeskBot(cfg, voice=False)
    # 用 HashingEmbedder 替代真实 llama-server 嵌入（确定性、免网络）
    from ..rag.embedder import HashingEmbedder
    dim = bot.embedder.dim
    hashing = HashingEmbedder(dim)
    if not hasattr(hashing, "embed_one"):
        def embed_one(text: str):
            return hashing.embed([text])[0]
        hashing.embed_one = embed_one  # type: ignore[attr-defined]
    bot.embedder = hashing
    bot.memory.embedder = hashing
    # 关闭记忆自动抽取（避免 FakeLlm.complete 被调 + 隔离每场景状态）
    bot.memory.extract_and_store = _async_noop  # type: ignore[method-assign]
    # 视觉关
    bot._maybe_vision = _async_noop  # type: ignore[method-assign]
    bot.rules.format = lambda: ""  # type: ignore[method-assign]
    # 反馈路径指向一个隔离的内存 collector（不写磁盘）
    bot.collector = FeedbackCollector(cfg)
    return bot


async def _async_noop(*args, **kwargs):
    return ""


async def _run_scenario(bot: DeskBot, fake: FakeLlm, scenario: Dict) -> Dict:
    """执行单个场景，返回 {passed, actual, branch, notes}。"""
    name = scenario["name"]
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            await bot._handle(scenario["input"])
        output = out.getvalue()
    except KeyboardInterrupt:
        output = out.getvalue() + "\n<KeyboardInterrupt>"

    # 期望分支判定：收集断言项
    checks = []
    for k, v in scenario.get("expect", {}).items():
        checks.append((k, v))
    results = _assert(scenario, output, bot, fake, checks)
    passed = all(r["ok"] for r in results)
    notes = "; ".join(r.get("note", "") for r in results if r.get("note"))
    branch = scenario.get("branch", "?")
    return {"name": name, "input": scenario["input"], "expected": scenario.get("expected", ""),
            "actual": output.strip()[:120], "passed": passed,
            "branch": branch, "notes": notes or None}


def _assert(scenario: Dict, output: str, bot: DeskBot,
            fake: FakeLlm, checks: List[tuple]) -> List[Dict]:
    """按场景的 expect 规则逐项断言，返回每项 {ok, note}。"""
    results: List[Dict] = []
    for key, val in checks:
        ok = True
        note = ""
        if key == "reply_contains":
            ok = val in output
            note = "" if ok else f"回答缺「{val}」"
        elif key == "no_reply_contains":
            ok = val not in output
            note = "" if ok else f"回答不应含「{val}」"
        elif key == "llm_called":
            ok = len(fake.calls) > 0
            note = "" if ok else "应调用 LLM 但未调"
        elif key == "llm_not_called":
            ok = len(fake.calls) == 0
            note = "" if ok else "不应调用 LLM 却调了"
        elif key == "llm_user":
            ok = fake.calls and fake.calls[-1]["user"] == val
            note = "" if ok else f"LLM user 应为「{val}」实为 {fake.calls[-1]['user'] if fake.calls else '无'}"
        elif key == "llm_system_has_date":
            sys_prompt = fake.calls[-1]["rag"] + fake.calls[-1]["rules"] if fake.calls else ""
            ok = bool(fake.calls) and _DATE_RE.search(sys_prompt)
            note = "" if ok else "system 未含今天日期"
        elif key == "store_has":
            ok = val in bot.store.all_texts()
            note = "" if ok else f"store 缺「{val}」"
        elif key == "store_source":
            ok = bot.store.has_source(val)
            note = "" if ok else f"store 缺 source={val}"
        elif key == "llm_rag_has":
            ok = bool(fake.calls) and val in fake.calls[-1]["rag"]
            note = "" if ok else f"RAG 上下文缺「{val}」"
        elif key == "raises":
            ok = "<KeyboardInterrupt>" in output
            note = "" if ok else "应退出(KeyboardInterrupt)"
        elif key == "is_noise":
            ok = _is_noise(scenario["input"]) is val
            note = "" if ok else "噪声判定不符"
        else:
            note = f"未知断言 {key}"
            ok = False
        results.append({"ok": ok, "key": key, "note": note})
    return results


def _reset_state(bot: DeskBot, fake: FakeLlm) -> None:
    """每场景前重置 store/llm/collector（多轮场景不调）。"""
    bot.store._reset()
    fake.reset()
    bot._current_task = None


async def run_dialogue(cfg: Config) -> Dict:
    """运行全部对话场景，返回报告 dict（含 details）。"""
    bot = _make_bot(cfg)
    fake = FakeLlm()
    bot.llm = fake

    SCENARIOS = [
        {"name": "wait_finish", "input": "今天天气怎么样", "branch": "正常问答",
         "expected": "完整回答（含日期上下文）", "expect": {"reply_contains": fake.default_answer,
                     "llm_called": True, "llm_user": "今天天气怎么样"}},
        {"name": "weather_city", "input": "明天杭州天气怎么样", "branch": "正常问答(非位置)",
         "expected": "走 LLM，非固定位置句", "expect": {"no_reply_contains": "记住你在",
                     "llm_called": True, "llm_user": "明天杭州天气怎么样"}},
        {"name": "memory", "input": "记住我喜欢吃火锅", "branch": "记忆",
         "expected": "固定句 + 入库 + 无 LLM",
         "expect": {"reply_contains": "好的，我记住了：我喜欢吃火锅",
                     "store_has": "我喜欢吃火锅", "llm_not_called": True}},
        {"name": "feedback_good", "input": "这个回答很好", "branch": "反馈好",
         "expected": "固定夸奖句 + 无 LLM",
         "expect": {"reply_contains": "谢谢夸奖，我会继续保持！", "llm_not_called": True}},
        {"name": "feedback_bad", "input": "回答错误", "branch": "反馈差",
         "expected": "固定抱歉句 + 无 LLM",
         "expect": {"reply_contains": "抱歉，我记住了，下次会改进。", "llm_not_called": True}},
        {"name": "location", "input": "我在杭州", "branch": "位置",
         "expected": "固定句 + 入库 + 无 LLM",
         "expect": {"reply_contains": "好的，我记住你在杭州。",
                     "store_source": "user_location", "llm_not_called": True}},
        {"name": "location_question", "input": "我在杭州吗？", "branch": "正常问答",
         "expected": "疑问句不提取位置，走 LLM",
         "expect": {"no_reply_contains": "记住你在", "llm_called": True}},
        {"name": "multi_turn", "input": "我叫小明", "branch": "记忆→问答",
         "expected": "第一句入库，第二句带多轮历史",
         "expect": {"store_has": "小明"}},
        {"name": "exit", "input": "退出", "branch": "退出",
         "expected": "KeyboardInterrupt", "expect": {"raises": True}},
        {"name": "noise", "input": "嗯", "branch": "噪声过滤",
         "expected": "_is_noise 判真", "expect": {"is_noise": True}},
        {"name": "location_placeholder", "input": "我在这里", "branch": "正常问答",
         "expected": "占位词不提取位置，走 LLM",
         "expect": {"no_reply_contains": "记住你在", "llm_called": True}},
        {"name": "barge_in", "input": "(音频层机制)", "branch": "插话打断",
         "expected": "_barge_in 取消任务 + stop 播放", "expect": {}, "scope": "audio-manual"},
    ]

    details = []
    for sc in SCENARIOS:
        _reset_state(bot, fake)
        # 多轮场景：执行两轮
        if sc["name"] == "multi_turn":
            details.append(await _run_multi_turn(bot, fake))
            continue
        if sc["name"] == "barge_in":
            details.append(await _run_barge_in(bot))
            continue
        details.append(await _run_scenario(bot, fake, sc))

    passed = sum(1 for d in details if d["passed"])
    total = len(details)
    report = {
        "ts": now_ts(), "total": total, "passed": passed,
        "score": round(passed / total, 3) if total else 0.0,
        "baseline_passed": _read_baseline(cfg), "details": details,
    }
    report_path = cfg.path("optimize.eval_report").parent / "dialogue_report.jsonl"
    append_jsonl(report_path, {k: v for k, v in report.items() if k != "details"})
    return report


async def _run_multi_turn(bot: DeskBot, fake: FakeLlm) -> Dict:
    """多轮场景：先记忆「我叫小明」，再问「我叫什么」。"""
    first = await _run_scenario(bot, fake,
                                {"name": "multi_turn_t1", "input": "我叫小明",
                                 "branch": "记忆", "expected": "入库",
                                 "expect": {"reply_contains": "好的，我记住了：小明",
                                             "store_has": "小明", "llm_not_called": True}})
    if not first["passed"]:
        first["name"] = "multi_turn"
        return first
    second = await _run_scenario(bot, fake,
                                 {"name": "multi_turn_t2", "input": "我叫什么",
                                  "branch": "正常问答", "expected": "记忆经 RAG 注入",
                                  "expect": {"llm_called": True, "llm_rag_has": "小明"}})
    second["name"] = "multi_turn"
    return second


async def _run_barge_in(bot: DeskBot) -> Dict:
    """插话打断机制测试：_barge_in 应 stop 播放 + 取消进行中任务。"""
    stop_called = {"n": 0}

    class FakeSpeaker:
        def stop(self):
            stop_called["n"] += 1

    bot.speaker = FakeSpeaker()  # type: ignore[assignment]

    task = asyncio.create_task(asyncio.sleep(30))
    bot._current_task = task
    await asyncio.sleep(0.01)  # 让任务进入运行态
    bot._barge_in()
    await asyncio.sleep(0)     # 让事件循环处理 cancel，cancelled() 才生效
    cancelled = task.cancelled()
    passed = stop_called["n"] == 1 and cancelled
    task.cancel()
    return {"name": "barge_in", "input": "(音频层机制)", "expected": "取消任务+stop播放",
            "actual": f"stop_calls={stop_called['n']}, cancelled={cancelled}",
            "passed": passed, "branch": "插话打断",
            "notes": "scope=audio-manual（机制验证，非真实音频流）" if passed
                     else f"失败: stop={stop_called['n']} cancelled={cancelled}"}


def _read_baseline(cfg: Config) -> int:
    from ..utils.state import read_jsonl
    rows = read_jsonl(cfg.path("optimize.eval_report").parent / "dialogue_report.jsonl", limit=1)
    return rows[-1].get("passed", -1) if rows else -1
