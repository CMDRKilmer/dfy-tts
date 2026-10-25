#!/usr/bin/env bash
# Prepare and verify the Raspberry-Pi half of the split TTS.
#
# The Pi speaks through this order: local cache -> LAN Qwen (via the reverse
# tunnel on 127.0.0.1:8765) -> local Piper. This script only checks that every
# prerequisite of that chain is in place; it changes nothing except the cache.
#
#   ./setup_pi.sh              # verify
#   ./setup_pi.sh --warm       # verify, then pre-synthesise the phrases below
#   ./setup_pi.sh --warm-only  # just warm the cache
#
# Exit status is 0 only when the local fallback chain is intact. The LAN half is
# reported as a warning, not a failure: a sleeping desktop is normal.

set -uo pipefail

VENV_PY=/home/pi/tts-venv/bin/python
VOICE=/home/pi/.local/share/dfy-tts/voice.onnx
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT="$SCRIPTS_DIR/tts_client.py"
SERVER="${DFY_TTS_SERVER:-http://127.0.0.1:8765}"

# Phrases worth having on disk: the ones a voice assistant repeats forever.
WARM_PHRASES=(
  "我在。"
  "好的，我知道了。"
  "这个问题本鲸鱼娘得想想。"
  "米饭煮好了哦。"
  "先别急，等我一下。"
  "抱歉，这个我做不到。"
)

fails=0
warns=0

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fails=$((fails + 1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; warns=$((warns + 1)); }

echo "== interpreter =="
if [ -x "$VENV_PY" ]; then
  ok "venv python: $($VENV_PY --version 2>&1)"
  for mod in piper onnxruntime soundfile; do
    if "$VENV_PY" -c "import $mod" 2>/dev/null; then ok "python module: $mod"; else bad "missing python module: $mod"; fi
  done
else
  bad "no venv at $VENV_PY -- create it with: python3 -m venv /home/pi/tts-venv && /home/pi/tts-venv/bin/pip install piper-tts onnxruntime soundfile"
fi

echo "== voice model =="
if [ -f "$VOICE" ]; then
  ok "onnx: $VOICE ($(du -h "$VOICE" | cut -f1))"
  [ -f "$VOICE.json" ] && ok "config: $VOICE.json" || bad "missing $VOICE.json (Piper needs it next to the model)"
else
  bad "missing $VOICE -- copy one from the desktop, e.g. scp desktop:tts/data/ref_voice/zh_CN-huayan-medium.onnx {,.json} to $(dirname "$VOICE")/voice.onnx"
fi

echo "== audio output =="
if id -nG | tr ' ' '\n' | grep -qx audio; then
  ok "user is in the audio group"
else
  warn "user not in the audio group in THIS session (run: sudo usermod -aG audio \$USER, then re-login; 'sg audio -c ...' works meanwhile)"
fi
if aplay -l >/dev/null 2>&1; then
  ok "aplay sees $(aplay -l 2>/dev/null | grep -c '^card') card(s)"
else
  warn "aplay cannot list sound cards here (missing group membership or no ALSA device)"
fi
[ -f "$HOME/.asoundrc" ] && ok "~/.asoundrc sets a default ALSA device" \
                         || warn "no ~/.asoundrc -- players may need an explicit -D plughw:Headphones"
command -v aplay >/dev/null && ok "player: aplay" \
                             || { command -v ffplay >/dev/null && ok "player: ffplay" || bad "no aplay and no ffplay"; }

echo "== LAN server =="
if "$VENV_PY" - "$SERVER" <<'PY' 2>/dev/null
import json, sys, urllib.request
url = sys.argv[1]
with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
    info = json.loads(resp.read())
print(f"{info['dtype']} {info['device']} loaded in {info['load_seconds']}s, calls={info['calls']}, hits={info['cache_hits']}")
PY
then
  ok "reachable at $SERVER"
else
  warn "not reachable at $SERVER -- is start_tts_lan.ps1 running on the desktop? (local Piper still works)"
fi

echo "== cache =="
"$VENV_PY" "$CLIENT" --stats 2>/dev/null | sed 's/^/  /' || warn "cache directory not readable"

if [ "${1:-}" = "--warm" ] || [ "${1:-}" = "--warm-only" ]; then
  echo "== warming ${#WARM_PHRASES[@]} phrases =="
  for phrase in "${WARM_PHRASES[@]}"; do
    if "$VENV_PY" "$CLIENT" --server "$SERVER" --warm "$phrase" 2>&1 | sed 's/^/  /'; then :; fi
  done
  "$VENV_PY" "$CLIENT" --stats | sed 's/^/  /'
fi

echo
if [ "$fails" -gt 0 ]; then
  echo "RESULT: $fails blocking problem(s), $warns warning(s)."
  exit 1
fi
echo "RESULT: local fallback ready, $warns warning(s)."
