# vision-detection：当前实现与交接（2026-09-16）

## 工作规则与实际部署
- 用户授权跨六项目协作；大量未提交实现不可擅自清理或回滚。不创建子代理，除非用户明确要求。
- 修改前检查并停止本机受管服务，不擅自打开摄像头/麦克风或播放。远端162的Qwen为常驻服务，不能随本机stop_all误停。
- 机器人名称“小圆”。用户要求自然多人对话、旁人记忆、视听身份、注意力约束打断、主动搭话。声纹/视觉冲突优先可靠声纹归属发言，不据此改写冲突人脸。
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

## 未识别的人也要保持同一个人：短期人脸重识别（2026-09-16，最新）
现场日志里两分钟内出现 anonymous_0001…0013 和反复的 unstable_face_body_voice_association，
但同一时间房间里只有一个访客。根因在这里：人脸跟踪是按框重叠做的（IoU<0.28 或断开 2 秒就换 track），
而 stable_person_id 在没认出用户时直接用 vision_<face_track_id>，所以人一转头、检测漏两帧，
下游就收到一个全新的人，注意力再建一个新的陌生人，这个人身上什么都攒不起来。
新增 GuestGallery：认不出是谁时，用已有的人脸模板（质量门槛 + 多帧均值那套，不是单帧）在会话内做重识别，
同一张脸拿回同一个 guest id，person_id 变成 vision_guest_0001，跟踪断了也不变。
- 阈值 0.45（VISION_GUEST_THRESHOLD），比登记匹配的 0.38 严：把两个陌生人合并成一个会把甲说的话算到乙头上，
  而把一个人拆成两个只是丢连续性，前者更糟。这个数字没有在真人脸上标定过。
- 10 分钟没再出现就忘掉（VISION_GUEST_TTL_S），最多 32 个人，不落盘、不是登记，人走了就没了。
- 匹配上之后模板按 0.9/0.1 缓慢更新，跟着光线和角度走，但一帧拉不动身份。
- 完全没拿到人脸嵌入（质量不过关）时仍然退回 vision_<face_track_id>，不假装认识。
新增用例：跟踪断了再回来还是同一个 guest、不同的脸是不同的 guest、过了 TTL 就是新人、
person_id 的优先级是 名字 > guest > 框跟踪。
验证：vision 31、attention 186、voice 84、brain 77 全过。
