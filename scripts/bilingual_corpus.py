"""Bilingual (zh_CN + en_US) material for the MiniMax corpus.

Background -- why this file exists at all.

A Piper Chinese voice ships an espeak ``cmn`` phonemizer. That voice's phoneme
table happens to be a *superset* of the English IPA inventory: probing
``zh_CN-huayan-medium`` showed its 152 phoneme keys already contain every one of
the 45 phonemes ``espeak('cmn')`` emits for English text and every one of the 49
phonemes ``espeak('en-us')`` emits. So a bilingual model needs **no** merged
symbol table and **no** re-mapped ids -- ``num_symbols`` stays 256 and the
checkpoint weights load unchanged.

What is *not* free: the training-time phonemizer and the ONNX-time phonemizer
differ. ``piper.phonemize_espeak`` (training) maps text with espeak and handles
Latin words happily; ``piper.phonemize_chinese`` (ONNX export) runs g2pW, which
only knows Chinese and silently drops every Latin character. Training on English
therefore produces a model whose ONNX build can never say the English it was
trained on unless the runtime phonemizer is replaced -- see
``scripts/phonemize_bilingual.py``.

Text normalisation rules used by every line below:

* numbers are spelled out (``twenty four``, ``seven thirty``), never digits;
* abbreviations are spaced single capitals (``W I F I``, ``U S B``) so espeak
  reads them letter by letter instead of guessing a word;
* English words stay lower case;
* each line is one short utterance, 8-25 Chinese characters or the English
  equivalent, so the same length window applies to both halves.

The split follows the reference design: ~40 % pure Chinese, ~40 % pure English,
~20 % code-switched.
"""

from __future__ import annotations

#: Pure English smart-home and assistant lines. Digit-free, spaced capitals.
ENGLISH = """Good morning, it is seven thirty in the morning.
The current temperature is twenty four degrees celsius.
The living room light has been turned on.
Curtain in the bedroom has been pulled closed.
Your alarm is set for six fifteen tomorrow.
Air conditioner is now set to twenty six degrees.
Humidity in the kitchen is a little high today.
Would you like me to dim the bedroom lamp?
The washing machine will finish in twenty minutes.
I could not reach the smart plug in the kitchen.
The front door was unlocked five minutes ago.
All devices in the house are now in standby.
The weather forecast says it will rain this afternoon.
Please remember to bring an umbrella with you.
There are three new messages waiting for you.
Your phone battery is low, please plug it in.
The network connection was restored just a moment ago.
Music is now playing at a lower volume.
Everything you did not need is now off.
Do you want me to read the news headlines aloud?
The vacuum robot has finished cleaning the hall.
The bedroom window is closed and locked for the night.
It is already nine o'clock, time to get ready.
The coffee machine will start in five minutes.
Tell me if you need anything else tonight.
The timer will ring in about thirty seconds.
The hallway light will turn off in two minutes.
Your afternoon meeting starts at half past two.
The outside air quality is good enough to open a window.
The laundry finished about ten minutes ago.
The smart lock reported a low battery level.
A shopping list just arrived on your phone.
The study light is still on upstairs.
Should I switch the air conditioner to sleep mode?
The indoor temperature dropped two degrees all of a sudden.
The doorbell rang, but nobody was waiting outside.
The radio will play soft music after nine.
The curtains will open automatically once the sun rises.
All the lights turned on at exactly seven.
Your reminder for the dentist is tomorrow morning.
Heater is warming the room to twenty degrees.
I found two devices that are currently offline.
The bathroom fan has been running too long.
The television will power off in one minute.
The humidity sensor in the basement is broken.
Please confirm whether I should lock the door.
Night lamp is on at the lowest brightness.
Your package will arrive between three and five.
The system update is ready to be installed.
Wake up call is set for six forty tomorrow.
Ping me when you are back home safely.
The guest room light has never been used today.
There is a strange sound coming from the fridge.
The smoke alarm tested fine this morning.
Your smart speaker is now on the network.
The sunlight sensor suggests the blinds should close.
Both windows upstairs are open and it is raining.
The water heater has reached sixty degrees already.
Doorbell is muted for the next hour.
The robot vacuum needs its filter replaced soon.
The app shows the bedroom is warmer than usual.
Everything is ready for your video call now.
The air purifier has run for six hours.
Your morning routine starts in about fifteen minutes.
The brightness is now at forty percent.
The garage door was opened and then closed.
The noise level outside is quite low tonight.
I turned the bedside lamp on for ten minutes.
The cooler in the study was switched off automatically.
Rain is expected again later on this evening.
Please tell me the weather in Beijing tomorrow.
The kitchen light flickered twice just now.
No movement was detected by the hallway motion sensor.
The lights are scheduled to dim at eleven.
The plant sensor says your fern needs water.
Charging for the smart watch ended a moment ago.
Room got warm, so the heater turned off.
The outside temperature fell below freezing overnight.
Someone pressed the doorbell twice and then left.
The fan speed has been lowered to the minimum.
Right now the living room is twenty three degrees.
The security camera captured nothing unusual last night.
The blinds in the office are fully closed now.
The laundry will be ready in about forty minutes.
Timer in the kitchen still has twelve minutes left.
A meeting reminder will come ten minutes early.
This month the air conditioner used less energy.
A firmware update is waiting for the smart lamp.
It feels dry in the bedroom, humidity is under forty percent.
After five minutes the bathroom light turned off.
The speaker will read your schedule out loud.
The alarm clock will not go off tomorrow morning.
The refrigerator door was left open again.
All four sensors are reporting normally right now.
The dishwasher finished its cycle ten minutes ago.
Night mode started once the sun went down.
The outdoor lights will turn on in thirty minutes.
The sound of rain is being played softly.
The kitchen lights are brighter than they were yesterday.
No devices reported any fault overnight.
The thermostat will stay at twenty two degrees.
Your evening reading lamp is ready to turn on.
The sensor detected a window being opened upstairs."""

#: Code-switched lines: a Chinese frame with English device names / terms.
MIXED = """已经把客厅的 air conditioner 打开了。
厨房的 smart plug 好像断开连接了。
提醒你，下午两点有一个 important meeting。
请把卧室的 W I F I 帮我重新打开。
已经帮你连上 U S B，可以拔了。
A I 助手正在等你说话。
先切换到 watch T V 模式吧，我投个屏。
蓝牙已经打开了，可以连耳机。
今天的 update 下载完了，要现在装吗。
收到一封来自 V I P 客户的 email。
把 bedroom 的温度调高两度。
客厅的 speaker 现在没有声音。
系统提示 low battery，记得充电。
这个 app 的界面换新了，看着还行。
卧室的 air purifier 该换滤芯了。
请帮我把 living room 的灯关掉。
他的 laptop 一直连着外网。
路由器重启之后 W I F I 就快多了。
今晚的 dinner 已经点好了。
明天早上七点，给我一个 alarm。
车库的 door 好像没关紧。
请把 music 的音量调小一点。
这份 report 需要今天交上去。
沙发旁边的 lamp 好像坏了。
别忘了提醒我晚上八点查 email 回信。
我把 schedule 重新排了一遍。
厨房的 fridge 温度有点高。
请把 study 的窗帘全部拉上，光太刺眼。
上个月的 bill 已经付过了。
手机提示 storage 快满了，得清一清。
把 speaker 的麦克风关掉吧。
他说 meeting 要往后推半小时。
卧室的 curtain 卡住不动了。
这台 printer 又连不上了，你去看一眼。
我把买菜的 list 发到你手机上了。
定时器已经设好了，三十分钟后响。
客厅的 sensor 检测到有人走动。
请把空调切到 sleep 模式。
今天的新闻 headline 我念给你听。
这盏 lamp 的亮度调到了四成。"""

#: Chinese wording for concepts the English lines also cover, so the two
#: languages stay balanced rather than the Chinese half dominating.
ZH_ONLY = """客厅的灯已经关了，你可以休息了。
卧室的空调调到二十六度吧。
明早七点叫我起床，别忘了。
如果下雨，记得收衣服。
厨房的抽油烟机还没关呢。
客厅的窗帘拉上，阳光太强了。
洗衣机再跑二十分钟就结束。
门口的门铃刚才响了两声。
书房的台灯亮度调低一点。
家里的设备现在都在待机。
空气质量不错，可以开窗。
扫地机器人把客厅扫完了。
卧室的加湿器该加水了。
热水器的温度已经够了。
走廊的灯过了两分钟自动灭。
冰箱的门又没关严。
把电视的声音调小一些。
阳台的晾衣架收回来了吗？
今晚不用开了，早点睡吧。
楼下的快递已经放在柜子里了。"""
