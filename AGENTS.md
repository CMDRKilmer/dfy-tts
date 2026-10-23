# AGENTS.md

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues in `CMDRKilmer/dfy-tts` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles map to same-named labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

single-context — one shared `GLOSSARY.md` at the repo root plus root `docs/adr/` for decisions. See `docs/agents/domain.md`.

## Repository layout

- `scripts/` — Windows training toolchain. `build_dataset.py` (audio → dataset), `train.py` /
  `train_voice.py` (training), `export_onnx_safe.py` / `export_voice.py` (ONNX export),
  `patch_monotonic_align.py` + `monotonic_align_core.py` (numba replacement for the missing
  Cython extension), `test_monotonic_align.py` (correctness test), `make_smoke_data.py`
  (synthetic smoke data), `setup.ps1` (bootstrap).
- Corpus tooling: `make_corpus.py` (writes `data/text/*.txt`, ten categories),
  `refdata_phonetic.py` (328 polyphonic + 115 tone-change lines), `bilingual_corpus.py`
  (English / code-switched / Chinese blocks), `make_minimax_jsonl.py` (MiniMax input),
  `check_corpus.py` (repeat / length / punctuation / ratio / phoneme-coverage audit).
- Bilingual runtime: `phonemize_bilingual.py` (routes Chinese to espeak `cmn`, Latin to
  `en-us`; `uncovered()` dry-runs a corpus), `say.py` (synthesis on PC and Pi),
  `test_phonemize_bilingual.py`.
- `.venv/` — Python 3.12 environment with `piper-tts[train]`, CUDA torch, onnxscript.
  Reinstalling `piper-tts` wipes the monotonic_align patch: re-run
  `scripts/patch_monotonic_align.py`.
- `data/` — datasets, caches, checkpoints, exported voices. Not committed.
- `README.md` — full walkthrough (setup, Windows pitfalls, MiniMax → fine-tune → Raspberry Pi).

## Environment facts (verified on this machine)

- Windows 11, no MSVC / Windows SDK / CMake / WSL / Docker. Do not propose C/C++ builds.
- RTX 2080 Ti, 11 GB, sm_75; driver 610.88 (CUDA 13.3 UMD) → use the `cu130` torch index.
- `uv` manages Python; `.venv/Scripts/python.exe` is the interpreter for every script here.
  The venv has **no pip and no uv module** — install with the global `uv pip install --python
  .venv/Scripts/python.exe <pkg>`.
- `python -m piper.train fit` alone breaks offline (two `ModelCheckpoint` callbacks, `val_mos`
  never logged) — use `scripts/train.py`.
- `--model.mos_metric none` unless the machine is online: UTMOS loads through `torch.hub`.
- The PyPI wheel needs `onnxscript` for ONNX export; it is not in the `[train]` extra,
  and `piper.phonemize_chinese` needs the `unicode-rbnf` package.
- Bilingual is asymmetric: **training** works unchanged (the Chinese `phoneme_id_map` already
  carries every English IPA symbol — verified zero gaps), but **inference** must use
  `scripts/say.py`. Stock `python -m piper` phonemizes with g2pW, which silently drops Latin
  words, so a bilingual model sounds mute on its English lines when driven by the stock CLI.
- espeak's `cmn` voice reads Latin letters badly (`XZ` → "x-ed"); `say.py` deliberately sends
  Latin runs to `en-us` instead. `test_phonemize_bilingual.py` pins both behaviours.

## Sandbox traps (workspace-write mode)

The harness sandbox denies child pipes and some directory operations. Each symptom below was
diagnosed the hard way; the fixes are already in the scripts, but do not undo them.

- **`PermissionError: [WinError 5]` from `subprocess.check_output`** → `typeshed_client` shells
  out for `sys.path`, `jsonargparse` swallows the failure, every piper class resolves to **zero**
  parameters, and training dies with `No action for key "model.batch_size"`. `scripts/train.py`
  and `scripts/export_onnx_safe.py` pre-seed the resolver singleton instead.
- **Same error from `multiprocessing.connection.Pipe`** → DataLoader workers. Keep
  `--data.num_workers 0` (`train_voice.py` now defaults to it).
- **`PermissionError: [Errno 13]` writing inside a `tempfile.mkdtemp` directory** → the sandbox
  ACL on `mkdtemp` dirs denies writes while `os.makedirs` dirs work fine. `export_onnx_safe.py`
  swaps `mkdtemp` for `makedirs`.
- **`PermissionError` from `os.rmdir` in a `finally` block** → `onnx_ir` cleanup after a
  *successful* write. `export_onnx_safe.py` makes rmdir non-fatal.
- `uv` and `ensurepip` both fail here (uv cannot read its cache or query the interpreter, pip
  cannot write its temp dir). Install packages from a normal terminal, not from inside a tool call.
- Leftover `.bilingual.onnx.data.*` directories in the export folder cannot be deleted by a
  sandboxed process; they are inert and can be removed from a normal shell.
