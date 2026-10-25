# 树莓派语音：本机兜底 + 局域网 Qwen 主路

这套东西回答一个问题：**树莓派上能不能跑 Qwen3-TTS 0.6B？** 结论是跑不了，但也不需要
它跑——把「音色」留在有显卡的电脑上，把「随叫随到」留在树莓派上。

---

## 1. 为什么树莓派跑不了

Qwen3-TTS 是自回归 TTS：每 12 Hz 生成一个 token（`Qwen3-TTS-12Hz-0.6B-Base`），
一秒音频要解码 12 步，每步都要过一遍 0.6B 的 transformer。

Pi 4B 的硬件上限：

| 指标 | 实测/规格 | 对 0.6B 自回归解码意味着 |
| --- | --- | --- |
| 内存带宽 | 3.65 GB/s（`dd` 实测） | 0.6B fp32 ≈ 2.4 GB，每步要把大半个矩阵读一遍 |
| FP32 算力 | ~9.4 GFLOPS（4×A72 @1.5 GHz） | 每步矩阵乘 ~1.2 GFLOP，即 ~0.13 s/token |

粗算：`12 token/s ÷ 每 token 0.13 s ≈ RTF 1.5`，还没算 KV cache 读写、层归一化、
采样开销和 Python/ONNX 胶水。实际只会更慢，用 int8 量化摊平矩阵也救不回来——**瓶颈是
显存带宽，不是精度**。一句话：Pi 4 上跑 0.6B 自回归 TTS，RTF 在「几十到上百」量级，
离线批处理都要等几十秒一句。

结论：**Pi 侧只做 24 kHz WAV 的搬运工和兜底，不做模型推理。**

---

## 2. 架构

```
┌─────────────────────────┐         ┌──────────────────────────┐
│ PC (Windows, 2080 Ti)   │  ssh -R │ Raspberry Pi 4           │
│                         │  8765   │                          │
│ tts_server.py           │◄────────┤ tts_client.py            │
│  · Qwen3-TTS 0.6B Base  │  隧道   │  1. 本地缓存（0.3 s）     │
│  · 克隆参考音 clone_... │         │  2. LAN Qwen（主路）      │
│  · 只监听 127.0.0.1     │         │  3. 本地 Piper（兜底）    │
└─────────────────────────┘         └──────────┬───────────────┘
                                               │ aplay
                                           3.5 mm 耳机孔
```

三层解析顺序（`scripts/tts_client.py`）：

1. **本地缓存** `~/.cache/dfy-tts/<sha1>.lan.wav` —— 重复的句子一次文件读取就够。
2. **LAN Qwen** —— `http://127.0.0.1:8765/tts`，隧道另一端就是显卡。音色是训练好的克隆音。
3. **本地 Piper** —— `say.py` + 中英双语音素器。音质差一档，但完全离线、实时。

### 为什么用 SSH 反向隧道，而不是直连 192.168.100.2:8765

PC 上**没有管理员权限**，Windows 防火墙不让放行入站端口；`netsh advfirewall` 需要提权。
隧道反过来由 Pi 建立连接：

```
ssh -R 127.0.0.1:8765:127.0.0.1:8765 pi@192.168.100.16
```

隧道存活期间 Pi 的 `127.0.0.1:8765` 就是 PC 的 8765。PC 不监听任何对外的口子，
也不需要动防火墙。代价是 PC 必须能 SSH 到 Pi（已配置 `~/.ssh/pi_ed25519` 免密）。

### 缓存为什么按后端分开

`.lan.wav` 和 `.piper.wav` 是两个槽位，而不是共用一个：

- 桌面机睡眠时降级到 Piper，写进 `.piper.wav`；
- 早上桌面机回来，`.lan.wav` 仍然不存在 → 自动走主路重新合成；
- 如果共用一个槽位，坏音质会被永久钉死。

---

## 3. 实测（2026-10-25）

环境：RTX 2080 Ti 11 GB / torch 2.14.1+cu130 / fp32 / Python 3.12；Pi 4B，Python 3.14。

| 场景 | 耗时 | 备注 |
| --- | --- | --- |
| 服务冷启动到 `/health` 可服务 | 3.4 s | 模型已在本地磁盘缓存 |
| **缓存命中** | **0.32 s** | 端到端，含 aplay 启动 |
| Qwen 主路，4.3 s 音频 | 14.4 s（RTF 3.33） | 经隧道往返，稳定在 RTF 3.3–4.2 |
| Qwen 主路，1.9 s 音频 | 6.5 s（RTF 3.38） | |
| Piper 兜底，冷启动 | 11.5 s | 之后同一进程内 ~1 倍实时 |
| 隧道断开 → 降级到 Piper | +2 s 连接超时 | `CONNECT_TIMEOUT = 2.0` |

数字说明白了这套方案的性格：**固定语（问候、确认、提醒）快到几乎无感；即兴长句要等十几秒**。
AstrBot 这类场景大部分是固定语，缓存命中率决定体验。

---

## 4. 部署

### 4.1 PC 侧（一次性）

```powershell
# Python 环境按 README 的「一键安装」准备，另外装 qwen-tts：
uv pip install --python .venv/Scripts/python.exe qwen-tts fastapi uvicorn

# 模型缓存必须落在仓库里，别用默认的 %USERPROFILE%\.cache\huggingface（见坑 #3）
$env:HF_HOME = "$PWD\.cache\huggingface"

# 起服务 + 隧道 + 双守护
powershell -ExecutionPolicy Bypass -File scripts/start_tts_lan.ps1
```

`start_tts_lan.ps1` 参数（都有默认值）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Port` | `8765` | 两端共用端口 |
| `-Dtype` | `float32` | **别改 float16**，见坑 #1 |
| `-PiHost` | `pi@192.168.100.16` | 隧道目标 |
| `-SshKey` | `~\.ssh\pi_ed25519` | 免密私钥 |
| `-RefAudio` | `clone_source/clip0002.wav` | 克隆参考音 |
| `-RefText` | `开始计时三十秒，别走神。` | **必须与参考音逐字一致** |
| `-Install` | – | 注册「登录时启动」计划任务 `dfy-tts-lan` |

脚本是双守护：服务挂了重启服务，隧道断了重连隧道，两者互不牵连。

### 4.2 Pi 侧

```bash
# 系统依赖（一次性）
sudo apt install -y alsa-utils
sudo usermod -aG audio pi          # 之后必须重新登录，见坑 #6

# 环境自检（检查 venv / piper / onnxruntime / 音频 / 隧道 / 服务）
./scripts/setup_pi.sh

# 预合成常用句，把最贵的首句开销挪到后台
./scripts/setup_pi.sh --warm

# 说话
python scripts/tts_client.py "深夜了，早点睡哦"
echo "主人，米饭煮好了哦" | python scripts/tts_client.py
python scripts/tts_client.py -f out.wav "只落盘，不播放"
python scripts/tts_client.py --health      # 隧道通不通
python scripts/tts_client.py --stats       # 缓存多大
```

`tts_client.py` 只用标准库 `urllib` / `http.client`——Pi 的 `tts-venv` 里没有
`requests`，也不想为它装。

### 4.3 开机自启

PC 侧：`start_tts_lan.ps1 -Install` 注册计划任务（登录时触发）。

Pi 侧（可选，让语音助手随系统启动）：

```bash
cat > ~/.config/systemd/user/dfy-tts-warm.service <<'EOF'
[Unit]
Description=Warm dfy-tts cache
[Service]
Type=oneshot
ExecStart=/home/pi/dfy-tts/scripts/setup_pi.sh --warm-only
[Install]
WantedBy=default.target
EOF
systemctl --user enable --now dfy-tts-warm.service
```

---

## 5. 踩坑清单（都是真踩过才写下来的）

### 坑 1：fp16 直接崩

`--dtype float16` 加载不报错，第一次解码就 CUDA 断言退出：

```
Assertion `input[0] != 0` failed  (TensorCompare.cu:121)
```

Qwen3-TTS 的采样/比较算子对半精度不友好。**锁 fp32**，代价是多占 ~1.2 GB 显存，
换来稳定。0.6B fp32 在 11 GB 卡上毫无压力。

### 坑 2：个别句子跑满 2048 步、单句 45 s 超时

自回归解码默认 `max_new_tokens=2048`。多数句子几十步就出 EOS，但个别句子（实测
「先别急，等我一下。」）会一直不收敛，把上限跑满——单句从 6 s 变成 45 s+，客户端超时。

修法：按文本长度动态估算上限（每字约 12 Hz × 语速），再加一个硬顶。修复后同一句
7.8 s。**任何自回归 TTS 上生产都要设这道闸。**

### 坑 3：`HF_HOME` 默认路径没有写权限

不设 `HF_HOME` 时 huggingface 往 `C:\Users\<user>\.cache\huggingface` 写，抛
`PermissionError`。在 `start_tts_lan.ps1` 里把它指到仓库内 `.cache/huggingface`。

### 坑 4：本地只有 Base，没有 CustomVoice 实体

`Qwen3-TTS-0.6B-CustomVoice` 在本地缓存里只有 ref，实体没下全，加载即失败。
能用的是 **Base**——它支持 zero-shot 克隆：给一段参考音 + **逐字对应的文本**。
多档音色就用多个参考音，`/tts` 的 `speaker` 字段只是缓存 key 的名字。

### 坑 5：PC 开不了端口

无管理员权限 → 防火墙不放行 → Pi 直连 `192.168.100.2:8765` 超时。改走
`ssh -R` 反向隧道（见 §2）。副作用：PC 睡眠时隧道一起断，正好由 Piper 兜底。

### 坑 6：`usermod -aG audio` 之后当前会话仍没权限

`id` 还是旧组，`aplay` 报 `Cannot get card index` / `no soundcards found` /
`Permission denied`。这是**会话组没刷新**，不是设备坏了：

```bash
sg audio -c 'aplay -l'    # 验证
```

重新 SSH 登录即可拿到新组；脚本里想立刻生效就用 `sg audio -c '...'` 包一层。

### 坑 7：PowerShell 5.1 必须带 UTF-8 BOM

这台 PC 只有 `powershell.exe`（5.1），没有 `pwsh`。无 BOM 的 UTF-8 脚本会被按 GBK
解析，中文字符串变成乱码甚至语法错误。**用任何工具重写这个脚本都要把 BOM 补回去。**

### 坑 8：别用 `Invoke-RestMethod` 做健康检查

本机代理会拦 `Invoke-RestMethod`。脚本里统一用 `curl.exe`（Windows 自带的那个，
不是 `curl` 别名）。

### 坑 9：`ref_text` 和参考音不一致 → 念错内容

零样本克隆靠强制对齐把文本和音频对齐。多一句参考音里没有的话，模型会把多余部分
拼到输出开头，表现为「语速慢一半、念的是参考句」。判据看字/秒：
正常 4–5 字/秒，出错时约 2 字/秒。**修 `ref_text` 而不是修推理参数。**

---

## 6. 运维

- **日志**：`data/_logs/tts_server.log`（服务端每句一行：耗时、RTF、字节数）。
- **健康**：`curl http://127.0.0.1:8765/health` 或 `python scripts/tts_client.py --health`。
- **缓存**：默认 `~/.cache/dfy-tts/`；想看占用用 `--stats`，想重来直接删。
- **降级行为**：隧道断 → 2 s 内判定不可达 → 本地 Piper，全程无声无息，界面上只是音色变了。
  如果想强制走主路（失败就报错），用 `--prefer lan`。
- **换参考音**：改 `start_tts_lan.ps1` 的 `-RefAudio` / `-RefText`（逐字一致！），重启服务。
  缓存 key 里带 `speaker`，换音色不会串味。

## 7. 已知限制 / 后续

- **RTF 3.3–4.2 对交互式偏慢**。短期靠缓存（固定语 0.3 s）；中期值得做的：
  流式输出（首包 ~2 s 就能起播）、`torch.compile`、修好 fp16 的算子断言。
- 目前一次一句，没有分句流水线。长段落建议在客户端切句。
- 隧道依赖 PC 的 SSH 到 Pi，Pi 改 IP 要同步改 `-PiHost`。
