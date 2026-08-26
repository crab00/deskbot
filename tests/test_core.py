"""Deskbot 核心模块单元测试（不依赖真实模型/硬件）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deskbot.utils.config import load_config  # noqa: E402
from deskbot.rag.memory import MemoryService  # noqa: E402
from deskbot.rag.store import VectorStore  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    import yaml
    cfg_file = tmp_path / "config.yaml"
    data = {
        "paths": {"root": ".", "models": "models", "logs": "data/logs",
                  "feedback": "data/feedback", "memories": "data/memories",
                  "datasets": "data/datasets"},
        "llm": {"system_prompt": "你是小桌。", "max_tokens": 10},
        "optimize": {"memory_keywords": ["记住"], "feedback_keywords_good": ["很好"],
                     "feedback_keywords_bad": ["不对"]},
        "rag": {"top_k": 4, "max_memory_chars": 400},
    }
    cfg_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(cfg_file)


def test_config_path_resolution(cfg):
    p = cfg.path("paths.memories")
    assert p.is_absolute()
    assert str(cfg.get("llm.max_tokens")) == "10"


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(tmp_path / "vectors", dim=8)
    emb = np.zeros(8, dtype=np.float32)
    emb[0] = 1.0
    store.add("我家有只猫叫咪咪", emb, {"source": "test"})
    assert len(store) == 1
    hit = store.search(emb, top_k=1)
    assert hit and "咪咪" in hit[0]["text"]
    store.persist()

    store2 = VectorStore(tmp_path / "vectors", dim=8)
    assert len(store2) == 1
    assert "咪咪" in store2.all_texts()[0]


def test_memory_json_parse():
    assert MemoryService._parse_json_array('["事实1", "事实2"]') == ["事实1", "事实2"]
    assert MemoryService._parse_json_array('```json\n["a"]\n```') == ["a"]
    assert MemoryService._parse_json_array("无内容") == []
    assert MemoryService._parse_json_array('结果是 ["唯一事实"]，就这样') == ["唯一事实"]


def test_feedback_analysis(cfg):
    from deskbot.optimize.feedback import FeedbackCollector
    fb = FeedbackCollector(cfg)

    acts = fb.analyze("记住，我喜欢喝咖啡", "")
    assert any(a["type"] == "memory" for a in acts)

    # 疑问句不该触发记忆
    acts = fb.analyze("你能记住我的名字吗？我叫小明", "")
    assert not any(a["type"] == "memory" for a in acts)

    acts = fb.analyze("这个回答很好", "")
    assert any(a["type"] == "feedback" and a["rating"] == "good" for a in acts)

    acts = fb.analyze("这个回答不对", "")
    assert any(a["type"] == "feedback" and a["rating"] == "bad" for a in acts)


def test_segmenter_energy_vad():
    from deskbot.audio.vad import EnergyVAD, SpeechSegmenter
    seg = SpeechSegmenter(EnergyVAD(threshold_db=-40), after_silence=0.5,
                          max_seconds=5.0, chunk_size=512)
    sr = 16000
    # 0.4s 静音 + 1.0s 1kHz 方波/正弦 + 0.6s 静音
    silence = np.zeros(sr // 10 * 4, dtype=np.float32)
    t = np.arange(sr, dtype=np.float32) / sr
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t[:sr])
    result = None
    for block in [silence, tone, np.zeros(sr // 10 * 8, dtype=np.float32)]:
        r = seg.feed(block)
        if r is not None:
            result = r
    assert result is not None
    assert result.size > sr * 0.5  # 捕获了大部分语音


def test_only_feedback():
    from deskbot.main import DeskBot
    assert DeskBot._only_feedback("这个回答很好", [{"rating": "good"}]) is True
    assert DeskBot._only_feedback("回答不对，那正确做法是什么？",
                                  [{"rating": "bad"}]) is False
    assert DeskBot._only_feedback("这个回答很好，帮我算一下 3+5",
                                  [{"rating": "good"}]) is False
