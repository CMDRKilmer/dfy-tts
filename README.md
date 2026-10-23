# Piper 训练环境（Windows）

在 Windows 上搭建 [Piper](https://github.com/OHF-Voice/piper1-gpl) 的训练环境并训练中文音色，**不需要
MSVC / Windows SDK / CMake**。整条链路（生成数据 → 训练 → 导出 ONNX → 合成试听）都已在本机跑通。

## 为什么需要这个仓库

PyPI 上的 `piper-tts` wheel 只带了预编译的合成端（espeak-ng 核心等），**没有**训练所需的 Cython 扩展
`piper.train.vits.monotonic_align.monotonic_align.core`。官方要求在 Linux 上用
`build_monotonic_align.sh` 现场编译，而 Windows 上没有编译器时训练会直接报：

```
ModuleNotFoundError: No module named 'piper.train.vits.monotonic_align.monotonic_align'
```

本仓库用 **numba JIT** 写了该扩展的等价实现（`scripts/monotonic_align_core.py`，算法与上游
`core.pyx` 逐行对应，并用穷举法验证了最优性），再由 `scripts/patch_monotonic_align.py`
写进 site-packages。numba 本来就在训练依赖里（来自 librosa），所以不需要额外的系统组件。

另外两处上游在「离线 + 新环境」下会踩的坑，也在脚本里绕开了：

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| 默认挂了两个 `ModelCheckpoint` | 离线时 `val_mos` 不会写出，第 1 个 epoch 末尾抛 `MisconfigurationException` / `RuntimeError: Found more than one stateful callback...`，**一个 ckpt 都不落盘** | `scripts/train.py` 只保留 `val_mel` 那一个回调 |
| 导出 ONNX 的逆流有断言 | 训练不充分时 `transforms.py:174 assert (discriminant >= 0)` 直接失败 | `scripts/export_onnx_safe.py` 走无断言的 forward 分支 |

## 一键安装

```powershell
pwsh -File scripts/setup.ps1
```

脚本做五件事：建 `.venv`（Python 3.12）→ 装 `piper-tts[train]` → 把 CPU 版 torch 换成
CUDA 版 → 装 `onnxscript` → 打 monotonic_align 补丁并跑自检。

> 如果你在国内、直连 PyTorch 源很慢：先手动把
> `torch-2.14.1+cu130-cp312-cp312-win_amd64.whl`（约 1.9 GB）下载到仓库根目录，`setup.ps1`
> 会发现它并直接本地安装（`--no-deps`）。实测直连只有 ~170 KB/s，走代理分 8 段并发下载约 8 分钟。
> 已下载的 wheel 留在 `.cache/wheels/`（不入库），重装时也可以直接指过去。

## 手动安装

```powershell
uv venv .venv --python 3.12

uv pip install --python .venv/Scripts/python.exe "piper-tts[train]"

# PyPI 上同版本号默认给 CPU 版 torch，必须显式重装 CUDA 版
uv pip install --python .venv/Scripts/python.exe `
  --index-url https://download.pytorch.org/whl/cu130 --upgrade --reinstall "torch==2.14.1"

# ONNX 导出（torch>=2.9 起 torch.onnx 需要 onnxscript，不在 [train] 依赖里）
uv pip install --python .venv/Scripts/python.exe onnxscript

.venv/Scripts/python.exe scripts/patch_monotonic_align.py
```

> CUDA 索引要跟驱动匹配：驱动支持 CUDA 13.x 用 `cu130`，12.x 用 `cu126`。每个索引里可用的
> torch 版本不同，装之前先确认该索引存在对应版本的 `cp312-win_amd64` wheel。
> 训练依赖写的是 `torch>=2,<3`，`2.14.1+cu130` 满足要求。

## 验证

一条命令跑完整条链路（对齐补丁 → 语料审计 → 双语音素器测试 → 造数据集 → GPU 跑 1 个 batch
→ 导出 ONNX → 中/英/混排三种合成）：

```powershell
.venv/Scripts/python.exe scripts/smoke_test.py
#   -> all smoke checks passed
#   最后一次实测：onnx 2.2 MB；官方 CLI 中文 1.00 s，
#   say.py 合成 zh 1.16 s / en 1.30 s / mix 1.50 s
```

只想要快检查（不训练，约 20 秒）：`scripts/smoke_test.py --skip-train`。
它会验证「英文句和中英混排行能通过数据集过滤」——这正是以前会被静默丢掉的部分。

分步验证：

```powershell
# GPU
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   -> 2.14.1+cu130 True NVIDIA GeForce RTX 2080 Ti

# 对齐搜索（numba 补丁）与穷举最优解逐例比对
.venv/Scripts/python.exe scripts/test_monotonic_align.py
#   -> all monotonic_align checks passed

# 合成数据跑 2 个 epoch，确认 loss 下降且 ckpt 落盘
.venv/Scripts/python.exe scripts/make_smoke_data.py
.venv/Scripts/python.exe scripts/train.py fit `
  --trainer.accelerator gpu --trainer.precision 16-mixed --trainer.max_epochs 2 `
  --model.mos_metric none --data.csv_path data/smoke/metadata.csv `
  --data.audio_dir data/smoke/audio --data.cache_dir data/smoke/cache `
  --data.config_path data/smoke/config.json --data.voice_name smoke `
  --data.espeak_voice en-us --data.trim_silence false --data.batch_size 4 `
  --model.sample_rate 22050 --trainer.default_root_dir data/smoke/run
#   -> epoch=0-val_mel=1.8244.ckpt, epoch=1-val_mel=1.7925.ckpt, last.ckpt
```

> 注意：官方默认的 `python -m piper.train fit` 在离线机器上跑不完第 1 个 epoch（见上面那个表格），
> 请用 `scripts/train.py`。已生成的 smoke 产物在 `data/smoke/`（不入库），
> `data/smoke/export/` 里那个 ONNX 是拿 2 个 epoch 的模型导出的，音质只为「能出声」，不要当成品。

## 完整流程：MiniMax 生成数据 → 2080 Ti 微调 → 树莓派离线合成

### 1. 收集文本

仓库里已经生成好了一套：`data/text/`（**1380 句 / 22019 字，中英双语**，1380 句里 1380 句不重复、开头全不同）。

```powershell
.venv/Scripts/python.exe scripts/make_corpus.py        # 重新生成分类文本
.venv/Scripts/python.exe scripts/make_minimax_jsonl.py # 合成 MiniMax 的 JSONL
.venv/Scripts/python.exe scripts/check_corpus.py       # 重复度 + 长度 + 标点 + 音素覆盖审计
```

| 文件 | 内容 | 句数 |
| --- | --- | --- |
| `data/text/special.txt` | 时间 / 时长 / 数字读法（计时三十秒、九五折、百分之二点七五…） | 71 |
| `data/text/polyphonic.txt` | **多音字**（328 个目标字，每句让目标字出现两次以上，覆盖多种读音） | 328 |
| `data/text/tone_change.txt` | **音变六族**（轻声 / 儿化 / 上声连读 / 一不变调 / 啊的音变 / 形容词变调） | 115 |
| `data/text/numbers.txt` | 数字 / 时间 / 地点（气温、价格、时刻、人数、日期、省份城市、地址） | 181 |
| `data/text/home.txt` | 智能家居指令（九个房间 × 设备、场景模式、定时、播报、门锁） | 184 |
| `data/text/daily.txt` | 日常闲聊与陪伴对话 | 94 |
| `data/text/anime.txt` | 二次元口吻（傲娇、撒娇、感叹、害羞、生气） | 96 |
| `data/text/literary.txt` | 古诗与写景抒情（给语气和韵律一个参照） | 88 |
| `data/text/reading.txt` | 朗读 / 播报腔（整句、正式语速） | 69 |
| `data/text/mixed.txt` | **中英双语**（94 句纯英文智能家居 + 60 句中英夹杂） | 154 |
| `data/text/minimax_tts.jsonl` | 给 MiniMax 用的合并稿（JSONL），按上面顺序排列 | 1380 |

**不重复是硬规则**。第一版是按房间 × 设备、数字 × 句式组合出来的，结果「今天气温X度，出门记得
带件外套。」出现二十一次——音色学到的是句式，不是语言。现在：

- 每个模板只出现一次：数字、气温全部配**各不相同**的句子（`今天气温一度…` 这类能自然接上的才好留下，
  接不上的组合——`车里气温有多低`、`一早我查了一下`——在生成时就被筛掉了）；
- 同义场景换写法：卧室的灯说了三句，是三件不同的事；二十个城市天气，二十种问法；
- 英文句互相不撞开头（比前两个词），中英混排的框架也不撞；
- 审计脚本 `scripts/check_corpus.py` 卡八条：**没有任何两行相同**、**没有任何两行开头相同**、
  **单个分类不超过全库 60%**、**行长在 8–25**（中文按字、纯英文按词）、**中文行以句末标点收尾**、
  **mixed 纯英文占比 30%–70%**、**没有字在超过 60 行里打转**、**每个音素都在模型的 `phoneme_id_map` 里**。
  最后一条最关键：它证明这批文本训练出来的模型**念得出每一个音**。

### 中文 + 英文（为什么能同时做）

Piper 的中文音色用 espeak-ng 的 `cmn` 音素表训练，而 `cmn` 输出的是 **IPA**，不是拼音。
中文音色配置里 `phoneme_id_map` 有 152 个符号、`num_symbols` 是 256——**空着 104 个 ID**，
而 12 句英文 + 6 句中英混排实测只用到 45 个音素，**全部已经在中文表里**（`æ` `ð` `θ` `ɚ` `ɾ` `ʒ` `ᵻ`…都在）。

所以：

- **训练端不用改任何东西**：不用合并音素表、不用改 `num_symbols`、权重直接加载。
  语料里写英文，训练时 `--data.espeak_voice cmn` 照常能读（它会用中式口音读英文单词）。
- **推理端必须换音素器**：官方 `python -m piper` 走 `piper/phonemize_chinese.py` 的 g2pW（中文转拼音），
  英文单词返回 `None`，查不到音素就**静默丢弃**——训练得再好也念不出英文。
  用 `scripts/say.py`（内部是 `scripts/phonemize_bilingual.py`）：中文走 espeak `cmn`（和训练完全一致），
  英文走 espeak `en-us`（自然英语），两段音素映射到同一张表。

```powershell
# 中英混排，一句直接合成
.venv/Scripts/python.exe scripts/say.py my.onnx "打开客厅的 lamp" -f out.wav

# 整份语料流式吐出 16-bit PCM（树莓派上是 aplay 一行）
.venv/Scripts/python.exe scripts/say.py my.onnx --pcm < lines.txt | aplay -r 22050 -f S16_LE
```

`--method chinese` 可以切回官方 g2pW 路径做对比。测试脚本：

```powershell
.venv/Scripts/python.exe scripts/test_phonemize_bilingual.py
#   -> ok  split preserves 6 strings
#   -> ok  Chinese phoneme parity on 300 lines   （和 espeak cmn 逐字对齐）
#   -> ok  english coverage on 143 lines, 0 missing symbols
#   -> all bilingual checks passed
```

`minimax_tts.jsonl` 每行是 MiniMax 语音合成接口的入参：

```json
{"text": "帮我打开客厅的灯。", "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"}}
```

顺序是先铺满刚需（时长/数字/多音字/音变/指令），越往后越是补充材料，所以 **只生成一部分也能直接用**：
`--limit 300` 就能截出前 300 句（约 9 分钟音频）。建议凑够 **30 分钟以上**，全部 1380 句大约 82 分钟。

> 中间一列的中英混排是按**字符**算长度（8–25），纯英文句按**词数**算（8–25 个词）——
> 25 个英文字符只有四个词，装不下一个句子。


### 2. 生成音频

三条路，产物格式一样（一个文件夹 + `texts.txt`），`build_dataset.py` 都吃。

**A. MiniMax（音色复刻，效果最好）**：用 MiniMax 音色复刻生成音频，导出到一个文件夹
（`wav/mp3/m4a/flac` 都行），文本用 `texts.txt`（`文件名|文本`）或与音频同名的 `.txt` 记录。

**B. Qwen3-TTS 零样本克隆（本地免费，推荐）**：拿一段 3 秒以上的参考音，用它克隆音色
再批量生成整份语料。**实测这把嗓子克隆出来 MFCC 音色距离 25**（对比现成音色里最像的
fanchen 是 77、huayan 是 175），耳朵能听出是同一把嗓子。

```powershell
uv pip install --python .venv/Scripts/python.exe -U qwen-tts

# 先 4 句验证克隆效果
.venv/Scripts/python.exe scripts/qwen_clone_generate.py `
  --ref-audio data/target/astrbot_real_voice.mp3 `
  --ref-text "哼！本小姐才不是大肥鱼呢，这是丰满的流线型身材。米饭煮好了哦，快来吃。" `
  --limit 4 --sample

# 批量生成 700 条（约 4.5 小时，可断点续传）
.venv/Scripts/python.exe scripts/qwen_clone_generate.py `
  --ref-audio data/target/astrbot_real_voice.mp3 `
  --ref-text "<参考音一字不差的文本>" `
  --limit 700 --sample --out data/clone_source
```

> **`--ref-text` 必须和参考音一字不差。** 它是脚本克隆音色的唯一依据，写错就会克隆出
> 另一个人的声音。脚本会先做校验：算出「字数 ÷ 秒数」，落在 3.0–7.5 字/秒之外直接拒绝
> 运行——这是能自动抓出的最常见错误。
>
> **`--sample` 同样不能省**，理由见下一条。

**C. Qwen3-TTS 预置音色（不需要参考音）**：

```powershell
.venv/Scripts/python.exe scripts/qwen_tts_generate.py --limit 60 --sample
```

> **`--sample` 不能省。** 不加它，脚本按 JSONL 顺序取前 N 条，而 `mixed`（所有英文都在那儿）
> 是最后一个分类——60 句会全是中文，英文一句都没有。加上 `--sample` 后按语言分层抽样，
> 默认 30% 是纯英文（`--english-share 0.3` 可调），60 句里就有 18 句英文。

**第一次跑要下载模型**：克隆用 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`（约 2.4 GB），预置音色用
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`（约 1.8 GB）。国内网络下 `huggingface_hub`
容易断，**下载一次成功后就别再让它联网**，改用本地快照路径：

```powershell
$LOCAL = (Get-ChildItem -Recurse -Directory "$PWD/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots" | Select-Object -First 1).FullName
$env:HF_HUB_OFFLINE = "1"
.venv/Scripts/python.exe scripts/qwen_clone_generate.py --ref-audio ... --ref-text ... --limit 700 --sample --model $LOCAL
```

产出的音频是 24–32 kHz 单声道；`build_dataset.py` 会转成 22050 Hz。

**实测数据**：

| 路线 | 数据量 | 训练 | 结果 |
| --- | --- | --- | --- |
| 预置音色 60 句 | 2.8 分钟 | 50 epoch / 13 分钟 | 链路通，**音色不像**（数据太少） |
| 克隆音色 700 句 | ~35 分钟 | 待跑 | 待验证 |

> **踩过的坑：Qwen 的预置音色音频不能直接喂 Piper。** 那批音频 8 kHz 以上残留 3.0% 能量
> （官方 huayan 是 0.0%）、RMS 0.050（官方 0.080），24k→22050 还有混叠。用它训出来的模型
> VAD 人声占比 **0%～23%**，全是杂音。低通 + 归一化清洗后也只到 23.5%。**克隆路线没这个问题**，
> 因为克隆输出直接继承了参考音的音质。

`--resume` 默认开着，中断后重跑只会补缺失的片段。

### 3. 转成 Piper 数据集

```powershell
.venv/Scripts/python.exe scripts/build_dataset.py --source path/to/minimax --out data/myvoice
```

输出 `data/myvoice/wavs/*.wav`（22050 Hz / 16 bit / 单声道）+ `metadata.csv`。脚本会
自动找 ffmpeg 转码（没有 ffmpeg 就退回 librosa），过滤超长/超短，并打印总时长。

**中英双语不会被丢**：只要转写里含汉字或英文字母就保留，中文行、纯英文行、中英混排行都过。
只有纯数字/纯符号那种念不出东西的行才会被跳过。真想把它们也留下，加 `--keep-non-text`。
合成音色建议凑够 **30 分钟以上**。

### 4. 训练

```powershell
# 先 1 个 batch 验路（不落盘）
.venv/Scripts/python.exe scripts/train_voice.py --dataset data/myvoice --voice-name zh_CN_minimax --fast-dev-run

# 真训；batch 8 + 16-mixed 在 11 GB 上很宽裕
.venv/Scripts/python.exe scripts/train_voice.py --dataset data/myvoice --voice-name zh_CN_minimax --epochs 800
```

`train_voice.py` 会补上 `gpu` / `16-mixed` / `mos_metric none` / 输出目录等本机需要的参数，其余
Lightning 参数可以在 `--` 之后透传（例如 `-- --trainer.limit_val_batches 8`）。
显存不够就调小 `--batch-size`，或用 `--accumulate 2` 补回等效 batch。

ckpt 落在 `data/myvoice/checkpoints/lightning_logs/version_*/checkpoints/`，`last.ckpt` 是最新的。

### 5. 导出 ONNX

```powershell
.venv/Scripts/python.exe scripts/export_voice.py `
  --checkpoint data/myvoice/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt `
  --config data/myvoice/config.json `
  --out data/myvoice-export --voice-name zh_CN_minimax
```

得到 `zh_CN_minimax.onnx` + `zh_CN_minimax.onnx.data`（权重，**必须一起拷**）+
`zh_CN_minimax.onnx.json`。先在电脑上试听：

```powershell
# 双语：中文、英文、中英混排都能念
.venv/Scripts/python.exe scripts/say.py data/myvoice-export/zh_CN_minimax.onnx `
  "你好，我是新音色。" "Good morning, it is seven thirty." "打开客厅的 lamp" -f test.wav
```

> **不要用官方 `python -m piper` 试听双语模型。** 它走的是 g2pW（中文转拼音）音素器，
> 英文单词会被**静默丢弃**——音频里那段直接没声音，也不报错。纯中文句子用官方命令没问题，
> 一旦有英文就必须用 `scripts/say.py`。

### 6. 部署到树莓派

```bash
pip install piper-tts
```

把 `.onnx`、`.onnx.data`、`.onnx.json` 三个文件拷过去，再把 `scripts/phonemize_bilingual.py`
和 `scripts/say.py` 一起拷过去（两个文件，无额外依赖）：

```bash
# 单句
python scripts/say.py zh_CN_minimax.onnx "打开客厅的 lamp" -f hello.wav

# 流式播放，边生成边出声（低延迟）
echo "主人，客厅的灯已经关好啦。" | python scripts/say.py zh_CN_minimax.onnx --pcm | aplay -r 22050 -f S16_LE
```

Pi 4 上 medium 模型实时合成没问题，Pi Zero 建议把 `--length-scale` 调到 1.1–1.2 稍微放慢，
或者用 `x_low` 档模型。

## 目录结构

```
scripts/
  setup.ps1                  # 一键装环境
  patch_monotonic_align.py   # 把 numba 实现写入 site-packages（幂等）
  monotonic_align_core.py    # maximum_path_c 的 numba 实现（核心补丁）
  test_monotonic_align.py    # 对齐算法正确性测试（与穷举最优解比对）
  make_smoke_data.py         # 生成合成 smoke 数据
  build_dataset.py           # MiniMax/任意音频 -> Piper 数据集（转码 + 过滤 + csv）
  make_corpus.py             # 生成训练文本（data/text/*.txt，10 个分类）
  refdata_phonetic.py        # 多音字 328 条 + 音变 115 条的素材
  bilingual_corpus.py        # 纯英文 103 句 + 中英混排 40 句 + 对照中文 20 句
  make_minimax_jsonl.py      # 合成 MiniMax 语音合成用的 JSONL
  qwen_tts_generate.py       # 用 Qwen3-TTS 本地生成语料音频（免费，用来跑通流程）
  phonemize_bilingual.py     # 中英分流音素器（推理端必需，含 uncovered() 干跑）
  test_phonemize_bilingual.py# 分段无损 / 中文奇偶校验 / 英文符号覆盖
  say.py                     # 双语合成器（电脑与树莓派都能用）
  train.py                   # 训练入口（修掉离线时 ckpt 不落盘的问题）
  train_voice.py             # 训练参数封装（gpu/16-mixed/目录约定）
  export_onnx_safe.py        # ONNX 导出（绕开欠训练的断言）
  export_voice.py            # ckpt -> onnx + onnx.json 的封装
  smoke_test.py              # 端到端自检（对齐/造数据/训练/导出/合成）
  check_corpus.py            # 语料审计（重复/长度/标点/比例/音素覆盖）
data/                        # 数据集与训练产物（不入库）
docs/agents/                 # 技能套件配置（issue tracker / 标签 / 域文档）
requirements.txt             # 已验证的依赖版本
```

重新安装 piper-tts 后补丁会丢失，重跑一次 `scripts/patch_monotonic_align.py` 即可。

## 已知限制

- numba 版 `maximum_path_c` 是 CPU 实现，只在该算子上替代编译扩展；模型本身仍然在 GPU 上训练。
  第一次调用有 JIT 编译开销（数秒），之后走缓存。
- 关闭 UTMOS（`--model.mos_metric none`）后没有 `val_mos` 指标；挑最终模型请听 `val_mel` 最好
  的几个 ckpt，或临时联网打开 UTMOS。
- **双语只在训练端天然成立，推理端必须换音素器**。官方 `python -m piper` 的 g2pW 会把英文静默丢弃；
  想保留英文就得用 `scripts/say.py`。如果你只能跑官方命令，那就只训练纯中文数据。
- **英文发音取决于底模**。中文音素表虽然覆盖了英文 IPA，但这个音色本身没听过英语，
  读英文会带口音。想让英文更地道，用 `--ckpt-path` 从一个英语底模起步、或训更多英文句。
- 首次联网加载 UTMOS 还要能访问 `torch.hub`（`~/.cache/torch/hub`）；纯离线机器一定加
  `--model.mos_metric none`，否则训练会额外尝试下载。
- `export_onnx_safe.py` 强制走 forward 分支；训练充分后想用上游原版导出也可以直接用
  `python -m piper.train.export_onnx`。
- 训练依赖 `torch>=2,<3`；升级 torch 时注意保持 CUDA 索引与驱动匹配。
- **Qwen3-TTS 只是流程验证工具**。60 句 / 2.6 分钟训出来的声音不像参考音色——数据量差了
  一个数量级（Piper 微调要 30 分钟以上）。另外 `qwen-tts` 会带进 `transformers` /
  `torchaudio`，装完记得复核 `torch.cuda.is_available()` 仍是 True（实测没被换掉）。
