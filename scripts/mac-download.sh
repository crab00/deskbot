#!/usr/bin/env bash
# ============================================================
# 【在 Mac 上运行】下载全部模型 + llama.cpp 预编译二进制
#
# 网络方案（全部直连，不用代理 —— 实测代理反而更慢）：
#   - Qwen GGUF      → ModelScope（国内，主），hf-mirror（备）
#   - ASR/TTS 模型   → hf-mirror（sherpa 官方 HF 仓库）
#   - llama-server   → conda-forge 预编译 linux-aarch64 CPU 版（Nano，免编译）
#   - llama-server   → macOS 用 Homebrew（brew install llama.cpp，自带 Metal）
#   - KWS 唤醒词模型 → GitHub release（k2-fsa/sherpa-onnx，走 gh 代理）
#   - Silero/YOLO    → 先试 hf-mirror，失败退 GitHub 代理（文件小，可接受）
#
# 用法：./scripts/mac-download.sh
# 产物：models/（含 Qwen3-0.6B 嵌入模型、KWS）+ third_party/llama-server.linux-aarch64
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"/{asr,tts,vad,llm,vision} "$ROOT/third_party"

# ---------- 下载器（断点续传 + 重试） ----------
fetch() { # fetch <name> <url> <outfile>
  local name="$1" url="$2" out="$3"
  [ -f "$out.ok" ] && { echo "  [跳过] $name"; return 0; }
  echo "  [下载] $name"
  if command -v wget >/dev/null 2>&1; then
    wget -c -q --show-progress --tries=20 --timeout=60 --waitretry=5 -O "$out" "$url" && touch "$out.ok"
  else
    # --speed-time/--speed-limit：持续 30s 低于 1KB/s 视为冻结，立即中止换重试
    curl -fL -C - --retry 20 --retry-delay 5 --retry-all-errors \
      --speed-time 30 --speed-limit 1000 \
      --progress-bar -o "$out" "$url" && touch "$out.ok"
  fi
}

ghurl() { # ghurl <github路径> —— 按速度优先选择代理
  for p in gh-proxy.com ghproxy.net ghfast.top; do
    if curl -sfI --max-time 8 "https://$p/$1" -o /dev/null 2>/dev/null; then
      echo "https://$p/$1"; return 0
    fi
  done
  echo "https://gh-proxy.com/$1"
}

echo "== 1/6 LLM: Qwen2.5-0.5B（默认，快，~490MB）+ 1.5B（备选，~1.1GB）=="
# 0.5B：速度优先（Nano 上生成 ~5.8 tok/s，1.5B 只有 ~1.4-3.5）。embed_dim 需配 896。
# 1.5B：质量优先，生成慢一倍多。embed_dim 需配 1536。
Q05_OUT="$MODELS/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"
[ ! -f "$Q05_OUT.ok" ] && {
  fetch "Qwen0.5B(ModelScope)" \
    "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "$Q05_OUT" || fetch "Qwen0.5B(hf-mirror)" \
    "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
    "$Q05_OUT"
  touch "$Q05_OUT.ok"
} || echo "  [跳过] Qwen0.5B 已就绪"
QWEN_OUT="$MODELS/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
QWEN_MS="https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf"
QWEN_HF="https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
if [ ! -f "$QWEN_OUT.ok" ]; then
  echo "  ModelScope 优先，失败自动切 hf-mirror"
  fetch "Qwen(ModelScope)" "$QWEN_MS" "$QWEN_OUT" || fetch "Qwen(hf-mirror)" "$QWEN_HF" "$QWEN_OUT"
  touch "$QWEN_OUT.ok"
else
  echo "  [跳过] Qwen1.5B 已就绪"
fi

# Qwen3-0.6B：Mac 端本地嵌入模型（RAG 用，embed_dim 1024；比 1.5B 小/快）
# 官方 Qwen/Qwen3-0.6B-GGUF 仅提供 Q8_0 量化（无 Q4_K_M）。
Q3_OUT="$MODELS/llm/Qwen3-0.6B-Q8_0.gguf"
Q3_MS="https://modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q8_0.gguf"
Q3_HF="https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
if [ ! -f "$Q3_OUT.ok" ]; then
  fetch "Qwen3-0.6B(ModelScope)" "$Q3_MS" "$Q3_OUT" || fetch "Qwen3-0.6B(hf-mirror)" "$Q3_HF" "$Q3_OUT"
  touch "$Q3_OUT.ok"
else
  echo "  [跳过] Qwen3-0.6B 已就绪"
fi

echo "== 2/8 ASR: SenseVoice-zh（默认，中文效果最佳，hf-mirror）=="
ASR_DIR="$MODELS/asr/sense-voice-zh"
mkdir -p "$ASR_DIR"
[ ! -f "$ASR_DIR/.ok" ] && {
  fetch "sense-voice model.int8.onnx" \
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx" \
    "$ASR_DIR/model.int8.onnx"
  fetch "sense-voice tokens.txt" \
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/tokens.txt" \
    "$ASR_DIR/tokens.txt"
  touch "$ASR_DIR/.ok"
} || echo "  [跳过] SenseVoice 已就绪"

# 备选：Paraformer-zh（引擎 auto/paraformer 时使用）
PARA_DIR="$MODELS/asr/paraformer-zh"
mkdir -p "$PARA_DIR"
[ ! -f "$PARA_DIR/.ok" ] && {
  fetch "paraformer model.int8.onnx" \
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/model.int8.onnx" \
    "$PARA_DIR/model.int8.onnx"
  fetch "paraformer tokens.txt" \
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/tokens.txt" \
    "$PARA_DIR/tokens.txt"
  touch "$PARA_DIR/.ok"
} || echo "  [跳过] Paraformer 已就绪"

echo "== 3/8 TTS: VITS-zh（多说话人，hf-mirror 逐文件）=="
TTS_DIR="$MODELS/tts/vits-zh-ll"
TTS_REPO="csukuangfj/sherpa-onnx-vits-zh-ll"
hf_file() { # hf_file <repo> <remote> <local>
  local repo="$1" remote="$2" local="$3"
  fetch "$remote" "https://hf-mirror.com/$repo/resolve/main/$remote" "$local"
}
if [ ! -f "$TTS_DIR/model.onnx" ]; then
  mkdir -p "$TTS_DIR/dict" "$TTS_DIR/dict/pos_dict"
  hf_file "$TTS_REPO" "model.onnx"  "$TTS_DIR/model.onnx"
  hf_file "$TTS_REPO" "tokens.txt"  "$TTS_DIR/tokens.txt"
  hf_file "$TTS_REPO" "lexicon.txt" "$TTS_DIR/lexicon.txt"
  for fst in date.fst new_heteronym.fst number.fst phone.fst; do
    hf_file "$TTS_REPO" "$fst" "$TTS_DIR/$fst"
  done
  for d in dict/hmm_model.utf8 dict/idf.utf8 dict/jieba.dict.utf8 \
           dict/stop_words.utf8 dict/user.dict.utf8 dict/README.md \
           dict/pos_dict/char_state_tab.utf8 dict/pos_dict/prob_emit.utf8 \
           dict/pos_dict/prob_start.utf8 dict/pos_dict/prob_trans.utf8; do
    hf_file "$TTS_REPO" "$d" "$TTS_DIR/$d"
  done
else
  echo "  [跳过] VITS 已就绪"
fi

# VAD: 用 sherpa-onnx 官方的 silero_vad.onnx（x/h/c 格式，经测试可靠）。
# 不要用 snakers4/silero-vad 的 v5 导出（带 sr 输入，且有坏 LSTM 分支）。
echo "== 4/8 VAD: Silero（sherpa 官方版，~628KB）=="
fetch "Silero VAD(sherpa)" "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
  "$MODELS/vad/silero_vad.onnx"

echo "== 5/8 视觉: YOLOv8n（~13MB，可选，失败不阻塞）=="
if [ ! -f "$MODELS/vision/yolov8n.onnx.ok" ]; then
  # 快速尝试两个版本号，都失败就跳过（M5 视觉可选）
  for tag in v8.3.0 v8.2.0 v8.1.0; do
    url="$(ghurl "https://github.com/ultralytics/assets/releases/download/$tag/yolov8n.onnx")"
    if curl -fL --max-time 120 --retry 2 --retry-delay 2 --progress-bar \
         -o "$MODELS/vision/yolov8n.onnx" "$url" 2>/dev/null; then
      touch "$MODELS/vision/yolov8n.onnx.ok"
      break
    fi
  done
  [ -f "$MODELS/vision/yolov8n.onnx.ok" ] || echo "  [跳过] YOLOv8n 下载失败（可稍后手动补，非必需）"
else
  echo "  [跳过] YOLOv8n 已就绪"
fi

echo "== 6/8 llama-server 预编译二进制（conda-forge linux-aarch64 CPU，.conda 格式，Nano 用）=="
LLAMA_PKG="third_party/llama.cpp.conda"
LLAMA_BIN="third_party/llama-server.linux-aarch64"
if [ ! -x "$LLAMA_BIN" ] || [ ! -d "$ROOT/third_party/llama-server-libs" ]; then
  fetch "llama.cpp(conda)" \
    "https://conda.anaconda.org/conda-forge/linux-aarch64/llama.cpp-10380-cpu_openblas_h12c65e4_0.conda" \
    "$LLAMA_PKG"
  rm -rf "$ROOT/third_party/llama-cpkg" && mkdir -p "$ROOT/third_party/llama-cpkg"
  unzip -q -o "$LLAMA_PKG" -d "$ROOT/third_party/llama-cpkg"
  PKG_TAR_ZST=$(ls "$ROOT/third_party/llama-cpkg"/pkg-*.tar.zst 2>/dev/null | head -1)
  if [ -z "$PKG_TAR_ZST" ]; then
    echo "  解包失败：未找到 pkg-*.tar.zst"
  else
    zstd -d -q -f "$PKG_TAR_ZST" -o "$ROOT/third_party/llama-cpkg/pkg.tar"
    tar -xf "$ROOT/third_party/llama-cpkg/pkg.tar" -C "$ROOT/third_party/llama-cpkg"
    cp "$ROOT/third_party/llama-cpkg/bin/llama-server" "$LLAMA_BIN"
    chmod +x "$LLAMA_BIN"
    # 运行时动态链接库（libllama.so / libggml-*.so），必须随二进制一起部署
    rm -rf "$ROOT/third_party/llama-server-libs"
    cp -r "$ROOT/third_party/llama-cpkg/lib" "$ROOT/third_party/llama-server-libs"
    echo "  已复制运行库 → third_party/llama-server-libs/"
  fi
else
  echo "  [跳过] llama-server 已就绪"
fi

echo "== 7/8 llama-server for macOS（darwin-arm64, Metal）=="
if command -v llama-server >/dev/null 2>&1; then
  echo "  [跳过] 已检测到 llama-server: $(command -v llama-server)"
  echo "         请确保 config.yaml 的 llm.server_path 指向它（Homebrew 默认 /opt/homebrew/bin/llama-server）"
elif command -v brew >/dev/null 2>&1; then
  echo "  未检测到 llama-server。请运行： brew install llama.cpp（自带 Metal）"
  echo "  装好后 config.yaml 的 llm.server_path 设为 $(brew --prefix llama.cpp)/bin/llama-server"
else
  echo "  未检测到 Homebrew。请安装 Homebrew 后： brew install llama.cpp，"
  echo "  或从 https://github.com/ggml-org/llama.cpp/releases 下载 macos-arm64 构建解包到 third_party/。"
fi

echo "== 8/8 KWS 唤醒词模型（WenetSpeech，中文，~20MB）=="
KWS_DIR="$MODELS/kws/wenetspeech-3.3M"
if [ ! -f "$KWS_DIR/tokens.txt" ]; then
  mkdir -p "$KWS_DIR"
  KWS_TARBALL="$MODELS/kws/wenetspeech.tar.bz2"
  KWS_URL="$(ghurl "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2")"
  fetch "KWS-WenetSpeech" "$KWS_URL" "$KWS_TARBALL"
  tar -xjf "$KWS_TARBALL" -C "$KWS_DIR" --strip-components=1
  echo "  已解包 KWS 模型 → $KWS_DIR"
else
  echo "  [跳过] KWS 模型已就绪"
fi

# 生成 keywords.txt（pinyin token；需已装 sherpa-onnx）
if [ -f "$KWS_DIR/tokens.txt" ] && [ ! -f "$MODELS/kws/keywords.txt" ]; then
  KW="${WAKE_KEYWORD:-小桌}"
  if command -v sherpa-onnx-cli >/dev/null 2>&1; then
    sherpa-onnx-cli text2token --tokens "$KWS_DIR/tokens.txt" --tokens-type ppinyin "$KW" > "$MODELS/kws/keywords.txt"
    echo "  已生成 keywords.txt（关键词: $KW）"
  else
    echo "  未检测到 sherpa-onnx-cli，跳过 keywords.txt 生成。装好后运行："
    echo "    sherpa-onnx-cli text2token --tokens $KWS_DIR/tokens.txt --tokens-type ppinyin \"$KW\" > models/kws/keywords.txt"
  fi
fi

echo
echo "=== 下载结果 ==="
MISS=0
for f in "$MODELS/llm/Qwen3-0.6B-Q8_0.gguf" \
         "$MODELS/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
         "$MODELS/asr/sense-voice-zh/tokens.txt" \
         "$MODELS/tts/vits-zh-ll/model.onnx" \
         "$MODELS/vad/silero_vad.onnx" \
         "$MODELS/kws/wenetspeech-3.3M/tokens.txt" \
         "$ROOT/third_party/llama-server.linux-aarch64"; do
  if [ -s "$f" ]; then echo "  ✓ $f ($(du -h "$f" | cut -f1))"; else echo "  ✗ 缺失 $f"; MISS=1; fi
done
[ "$MISS" = "0" ] && echo "模型就绪！Mac 端还需 brew install llama.cpp（Metal）；Nano 部署： ./scripts/dev-sync.sh crab@192.168.31.202"
exit "$MISS"
