# vision-detection：实现交接（2026-09-16）

## 工作规则与实际部署
- 用户授权跨六项目协作；大量未提交实现不可擅自清理或回滚。不创建子代理，除非用户明确要求。
- 修改前检查并停止本机受管服务，不擅自打开摄像头/麦克风或播放。远端162的Qwen为常驻服务，不能随本机stop_all误停。
- 机器人名称“小艾克斯”。用户要求自然多人对话、旁人记忆、视听身份、注意力约束打断、主动搭话。声纹/视觉冲突优先可靠声纹归属发言，不据此改写冲突人脸。
- 本机启动：cd ~/Golands/biomimetic-brain-test && bash scripts/start_all.sh；停止：bash scripts/stop_all.sh；状态：python3 scripts/mac_stack.py status。
- Python：~/Golands/robot-attention-perception/.venv-mac/bin/python。MediaPipe固定0.10.32，不盲目升级。
- 端口：Vision8080、Voice8090、Attention8092、Brain8094、Memory8788、ROS9090；慢脑http://172.16.60.162:18050/v1/chat/completions，model=qwen3.5-9b。
- 所有基准和单元测试仅说明对应样本/程序行为，不能描述为现场识别准确率达标。
- 不在代码/文档保存SSH密码。完整历史见voice-detection/AGENTS.md下方历史记录；本节当前状态优先于过期历史。
- /Applications/声音/Mac全系统启动与三项验收.txt保持“一键启动／分别启动”两块，详细验收独立保存。

## 本项目已实现
- 本地摄像头服务8080，YOLOv8n物体/人体、YuNet人脸关键点、SFace身份、FER+表情、MediaPipe Hand手势、Face Landmarker嘴唇几何。
- FER+输入改回0–255灰度尺度，低分/类别接近标无效；表情不等于真实心理，negative score仍工程映射。
- 唇动由归一化嘴唇开合时序判断，代替像素差；跟踪预热、关键点/角度/大小不可靠时lip_motion_valid=false。静止张嘴不算说话，咀嚼也可能嘴动。
- 注册连续同脸至少5样本、至少2帧有效唇动，模板一致性预检，dry_run支持与Voice两边先验证。正式身份存config/identities.local.json。
- 脸框不再写死stranger，显示实际角色；原始人物身份/表情/注视/唇动与测量时间和model_timings输出。
- reflex_area_ratio专用脸框，不混用人体框引起假靠近。物体检测含面积/中心/时间；FlashDetector记录短亮度上升回落事件，支持视听脉冲绑定。
- 模型下载脚本与Brain启动预检包含face_landmarker.task。

## 关键文件、验证及限制
- scripts/local_vision_dashboard.py、lip_landmarks.py、flash_events.py、download_models.sh。
- 最近Vision8项通过；真实静态样本Face Landmarker新增约9.5ms（样本非系统保证）。
- RGB尺寸距离是monocular_size_proxy，非实测3D；不是完整读唇模型，不存在可靠的人际眼神指向估计。未执行实体转头。

详细历史与最新追加记录请同时阅读本项目AGENTS.md，冲突以用户最新指令为准。

## 统一注意力竞争（第五版，2026-09-16 最新）
理论链已全部落地：Itti-Koch 显著性 → Biased Competition → Reynolds-Heeger 除性归一化（v2，归一化池跨模态耦合 κ=0.35）
→ Selective Tuning（v2 方位抑制环 + workspace soft-WTA 压制非赢家）→ Habituation/IOR/Hysteresis → Global Workspace → Brain。
唯一没做的是层级 Selective Tuning：候选空间是扁平的 person/track/object，没有真正的 part-whole 层级，硬套是空架子。
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（编译 bash scripts/build_attention_plugins.sh）：av_memory_language_v1（默认，自上而下相加）、
av_memory_language_v2（乘性注意场 + 跨模态池 + 抑制环）、builtin_audio_visual_v1（仅视听基线）。
v1/v2 跑同一套行为用例全过；现场优劣未比较，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
Global Workspace（global_workspace.py）是统一订阅点也是认知层竞争：soft-WTA(share=a²/Σa²) + 有限容量 4 +
urgency≥0.7 抢占；非感知候选（记忆、惊跳）在这里竞争，因为它们没有模态。
入口 GET /api/workspace、?since=N 回放、/api/workspace/stream (SSE)、ROS /attention/workspace。订阅者落后丢帧并计数。
双向：Memory→Attention 用 memory_candidates.py（保守：只提未完成的事、每 5 分钟一次、人在场、安静 3 秒、
同一件事只提一次，ATTENTION_MEMORY_EVENTS=0 可关）；Brain→Attention 用 attention_memory.goals 作 top-down bias
（awaiting_answer/holding_floor/deferred_turn/greet_owner，只偏置不放行）。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型：声纹 ERes2NetV2（192 维）；人脸走商用路线，默认 sface(Apache-2.0) + 质量门槛 + 5 帧模板均值，
arcface(buffalo_l) 已下载但权重仅限非商业研究，--face-backend arcface 可切；身份识别 300ms 节流。
Attention 164、Brain 62、Vision 23、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第四版历史，2026-09-16）
输入源五个（8092 可勾选，存 .run/attention-config.json）：audio_visual、memory_context、linguistic_context、
cross_session_memory、internal_state（默认关闭，无生产者）。关闭=消融该通道证据，不是关硬件或安全反射。
算法为独立 .so（C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh）：
av_memory_language_v1（默认，自上而下相加）、av_memory_language_v2（Reynolds-Heeger 乘性注意场 + Selective Tuning
方位抑制环，交流意愿层与 v1 共享 attention_common.hpp）、builtin_audio_visual_v1（内置仅视听基线）。
v1/v2 跑同一套行为用例且全过；两者优劣未经现场对比，所以默认仍是 v1。缺 .so 不阻断启动，回退基线并报错。
统一订阅点 global_workspace.py：每个感知周期广播一份内容（cycle/focus/target/engagement/coalition/sources/
algorithm/provenance）。入口 GET /api/workspace、/api/workspace?since=N 回放、/api/workspace/stream (SSE)、
ROS /attention/workspace。订阅者落后会丢帧并计数，不阻塞感知循环。Brain 现有 brain_input 通路未改。
结构：自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来接 arousal/motivation 与第二层通道。
记忆通道整体乘以识别置信度；角色（主人/陌生人）不进交流意愿层，只留在第一层 importance 影响「看哪里」。
模型已替换：声纹 CampPlus -> ERes2NetV2（192 维，3 秒语音 14.1->53.9ms，句末计算）；
人脸新增可切换后端，默认仍是 sface（Apache-2.0），arcface(buffalo_l w600k_r50) 已下载但权重仅限非商业研究，
需确认用途后用 --face-backend arcface 启用；身份识别加 300ms 节流。旧声纹/人脸档案已按指示清空，
备份在 ~/Golands/.identity-backup-*。阈值未在真人数据上标定。
Attention 151、Brain 62、Vision 16、C++ 4 个测试目标、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重与阈值是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第三版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板「注意力输入源与算法」可勾选来源、切换算法，选择存 .run/attention-config.json。
来源五个：audio_visual、memory_context（本会话工作记忆）、linguistic_context（当前语境文本）、
cross_session_memory（跨会话熟悉度，后台查 hri-memory-service，快循环只读缓存）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与「无交流意愿层」。
结构上是 自下而上（视听显著度、声学突变、句首唤醒词）+ 自上而下（工作记忆目标、熟悉度先验）的归一化竞争，
其上再加一层按人的序贯证据累积决定「是不是在跟我交流」。内部状态将来作为 arousal/motivation 接第一层、连续通道接第二层。
记忆通道整体乘以识别置信度：身份不可靠时那份历史不能替当前这个人说话；角色（主人/陌生人）不进入交流意愿层，
只保留在第一层 importance 影响「看哪里」，权限与内容仍归 Brain。
无声邀请、转头回答、自言自语后转向、称呼他人、熟人回访这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 143 项、Brain 62 项、C++ attention_plugin_test 9 项、页面 DOM 通过；
scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查（九个场景，含三组消融/对照）。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力（第二版历史，2026-09-16）
注意力输入源与算法都已解耦，8092 面板"注意力输入源与算法"可勾选来源、切换算法，选择存 .run/attention-config.json。
来源：audio_visual、memory_context、linguistic_context（当前语境文本，新增）、internal_state（默认关闭，无生产者）。
算法为独立 .so，C ABI 见 include/robot_attention_perception/attention_plugin_abi.h，编译 bash scripts/build_attention_plugins.sh：
av_memory_language_v1（默认，视听+记忆+语境，含按人累积的交流意愿层）、builtin_audio_visual_v1（内置仅视听对照基线）。
缺少 .so 不阻断启动，自动回退基线并在面板显示加载错误与"无交流意愿层"。
无声邀请、转头回答、自言自语后转向、称呼他人这些情况不再有各自的代码分支，由同一次融合更新产生。
Attention 132 项、Brain 66 项、C++ attention_plugin_test、页面 DOM 通过；scripts/verify_attention_algorithm.py 为不开硬件的端到端离线检查。
权重是工程先验，未做现场 ROC 标定；论文数字不能当本机准确率。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。

## 可插拔注意力输入（第一版历史，2026-09-16）
Attention现在有视听/记忆/可选内部状态支持源；本项目既有感知、记忆存储或模型服务职责不变，不能把“来源关闭”误作停硬件/安全通路。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。
