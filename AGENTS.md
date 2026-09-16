# vision-detection：当前实现与交接（2026-09-16）

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

## 可插拔注意力输入（2026-09-16）
Attention现在有视听/记忆/可选内部状态支持源；本项目既有感知、记忆存储或模型服务职责不变，不能把“来源关闭”误作停硬件/安全通路。
完整研究依据/协议/限制见 /Applications/声音/可插拔仿生注意力与论文依据.txt。没有内部状态也可试用第一版，但现场可靠性未验收，不能搬用论文准确率。

## 会话切换交接（2026-09-16，最新）
新AI先读 `/Users/letianxing/Golands/voice-detection/SESSION_HANDOFF.md`，再读本项目当前摘要。
本轮已核查：本机全部感知/Brain服务停止，162远端Qwen健康运行。最新Attention101/Brain60/DOM测试通过，不等于现场准确率。没有提交git，保护所有未提交修改和用户数据。
