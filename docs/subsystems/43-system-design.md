---
id: system-design
title: "System Design: End-to-End Case Studies"
status: complete
complexity: high
related:
  - "01-ego-collection.md"
  - "03-sim2real.md"
  - "04-teleoperation.md"
  - "../03-perception/02-depth-estimation.md"
  - "../03-perception/03-slam.md"
  - "../01-foundation/03-vla.md"
  - "../05-integration/01-pipeline-patterns.md"
  - "../05-integration/02-quality-gates.md"
prerequisites:
  - "Pipeline Patterns"
  - "Quality Gates"
  - "Data Collection Paradigms"
  - "Perception Algorithms"
  - "VLA Models"
last_validated: 2026-06-27
---

# System Design: 绔埌绔郴缁熻璁℃渚?

## 搂0 鈥?One-liner

涓変釜瀹屾暣鐨?embodied AI 绯荤粺鏋舵瀯妗堜緥鈥斺€斾粠鏁版嵁閲囬泦鍒版ā鍨嬮儴缃茬殑鍏ㄩ摼璺璁★紝娑电洊 Ego-centric銆丼im2Real銆侀仴鎿嶄綔涓夌鏍稿績鑼冨紡銆?

## 搂1 鈥?Concept Map

```mermaid
graph TB
    subgraph "Case 1: Ego-centric Pipeline"
        C1A[Ego閲囬泦] --> C1B[鎰熺煡澶勭悊]
        C1B --> C1C[鑷姩鏍囨敞]
        C1C --> C1D[VLA璁粌]
        C1D --> C1E[杈圭紭閮ㄧ讲]
    end

    subgraph "Case 2: Sim2Real Pipeline"
        C2A[浠跨湡鍣╙ --> C2B[鍩熼殢鏈哄寲]
        C2B --> C2C[绛栫暐璁粌]
        C2C --> C2D[鍩熼€傚簲]
        C2D --> C2E[鐪熷疄閮ㄧ讲]
    end

    subgraph "Case 3: Teleoperation Pipeline"
        C3A[閬ユ搷浣滈噰闆哴 --> C3B[鍔ㄤ綔瀵归綈]
        C3B --> C3C[VLA璁粌]
        C3C --> C3D[鑷富鎵ц]
    end
```

## 搂2 鈥?Case 1: Ego-centric 鏁版嵁閲囬泦鍒拌竟缂橀儴缃?

### 2.1 鍦烘櫙鎻忚堪

鏋勫缓涓€涓涓€浜虹О瑙嗚 (Ego-centric) 鐨勫帹鎴挎搷浣滃姪鎵嬬郴缁熴€傜敤鎴蜂僵鎴村ご鎴磋澶囨墽琛屾棩甯告搷浣滐紝绯荤粺閲囬泦鏁版嵁銆佽缁?VLA 妯″瀷锛屾渶缁堥儴缃插湪杈圭紭璁惧涓婃墽琛岃嚜涓绘搷浣溿€?

### 2.2 鏋舵瀯鍥?

```mermaid
graph TB
    subgraph "Collection Layer"
        HW1[澶存埓鐩告満<br/>RealSense D435i]
        HW2[鑵曢儴鐩告満<br/>GoPro Hero]
        HW3[IMU + 楹﹀厠椋嶿
        REC[ROS2 褰曞埗鑺傜偣]
        HW1 --> REC
        HW2 --> REC
        HW3 --> REC
    end

    subgraph "Perception Layer"
        SYNC[鏃堕棿鍚屾<br/>hardware_sync]
        CAL[鑱斿悎鏍囧畾<br/>Kalibr]
        DEPTH[娣卞害浼拌<br/>FoundationStereo]
        SLAM[SLAM<br/>ORB-SLAM3]
        HAND[鎵嬪Э浼拌<br/>HaMeR]
        SYNC --> CAL --> DEPTH
        CAL --> SLAM
        DEPTH --> HAND
    end

    subgraph "Annotation Layer"
        AUTO[VLM 鑷姩鏍囨敞<br/>GPT-4V / LLaVA]
        MANUAL[浜哄伐鏍￠獙<br/>CVAT]
        MERGE[鏍囩铻嶅悎<br/>澶氭簮鎶曠エ]
        AUTO --> MERGE
        MANUAL --> MERGE
    end

    subgraph "Training Layer"
        DATASET[鏁版嵁闆嗙粍瑁?br/>RLDS / LeRobot]
        TRAIN[VLA 璁粌<br/>OpenVLA / Octo]
        EVAL[璇勪及<br/>鎴愬姛鐜?/ 娉涘寲鎬
    end

    subgraph "Deployment Layer"
        OPT[妯″瀷浼樺寲<br/>TensorRT / ONNX]
        EDGE[杈圭紭璁惧<br/>Jetson AGX]
        EXEC[鎵ц鍣?br/>鏈烘鑷?/ 澶圭埅]
    end

    REC --> SYNC
    HAND --> AUTO
    SLAM --> AUTO
    MERGE --> DATASET
    DATASET --> TRAIN
    TRAIN --> EVAL
    EVAL --> OPT
    OPT --> EDGE --> EXEC
```

### 2.3 缁勪欢娓呭崟

| 缁勪欢 | 閫夊瀷 | 鑱岃矗 | 鏇夸唬鏂规 |
|------|------|------|----------|
| 澶存埓鐩告満 | Intel RealSense D435i | RGB-D + IMU 鍚屾閲囬泦 | ZED 2i, Azure Kinect |
| 鑵曢儴鐩告満 | GoPro Hero 11 | 鎵?鐗╀氦浜掔壒鍐?| Insta360 GO 3 |
| 鏃堕棿鍚屾 | PTP (Precision Time Protocol) | 浜氭绉掔骇澶氫紶鎰熷櫒鍚屾 | NTP + 杞欢鍚屾 |
| 鑱斿悎鏍囧畾 | Kalibr | 鐩告満-IMU 鏃剁┖鏍囧畾 | Basalt |
| 娣卞害浼拌 | FoundationStereo | 楂樼簿搴﹀弻鐩繁搴?| DepthPro, ZoeDepth |
| SLAM | ORB-SLAM3 | 瑙嗚-鎯€?SLAM | Kimera, OpenVINS |
| 鎵嬪Э浼拌 | HaMeR | 3D 鎵嬪Э閲嶅缓 | 100DOH, MediaPipe |
| 鑷姩鏍囨敞 | GPT-4V API | 鍦烘櫙鐞嗚В涓庡姩浣滄弿杩?| LLaVA, Qwen-VL |
| 浜哄伐鏍￠獙 | CVAT | 杈圭晫妗?鍏抽敭鐐规爣娉?| Label Studio, VGG |
| 璁粌妗嗘灦 | LeRobot | 鏈哄櫒浜哄涔犵粺涓€妗嗘灦 | RLDS, Robomimic |
| VLA 妯″瀷 | OpenVLA | 寮€婧?VLA 鍩哄骇 | Octo, RT-1/2 |
| 杈圭紭閮ㄧ讲 | TensorRT | 鎺ㄧ悊鍔犻€?| ONNX Runtime, TFLite |
| 璁＄畻骞冲彴 | Jetson AGX Orin | 杈圭紭鎺ㄧ悊 | Intel NUC, Raspberry Pi 5 |

### 2.4 鏁版嵁娴?

```
閲囬泦闃舵 (100 Hz IMU, 30 Hz RGB-D, 60 Hz 鑵曢儴)
    鈹?
    鈻?
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?ROS2 Topic                                  鈹?
鈹?/camera/color/image_raw     [sensor_msgs/Image]       30 Hz
鈹?/camera/depth/image_rect_raw [sensor_msgs/Image]      30 Hz
鈹?/camera/imu                  [sensor_msgs/Imu]        100 Hz
鈹?/wrist_camera/image_raw     [sensor_msgs/Image]       60 Hz
鈹?/audio                       [audio_common/AudioData] 16kHz
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
    鈹?
    鈻?
棰勫鐞?(鏃堕棿鍚屾 + 鍘荤暩鍙?+ 鍘嬬缉)
    鈹?
    鈻?
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹?鍚屾甯?(SyncedFrame)                        鈹?
鈹?timestamp: unix_ns                          鈹?
鈹?rgb: [H, W, 3] uint8                        鈹?
鈹?depth: [H, W] float32 (meter)               鈹?
鈹?imu: {accel: [3], gyro: [3]}                鈹?
鈹?wrist_rgb: [H, W, 3] uint8                 鈹?
鈹?pose: SE(3) camera-to-world                 鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
    鈹?
    鈹溾攢鈹€鈻?娣卞害浼拌 鈹€鈹€鈻?绋犲瘑娣卞害鍥?鈹€鈹€鈹?
    鈹?                              鈹?
    鈹溾攢鈹€鈻?SLAM 鈹€鈹€鈹€鈹€鈹€鈹€鈻?鐩告満杞ㄨ抗 鈹€鈹€鈹€鈹€鈹€鈹も攢鈹€鈻?鑷姩鏍囨敞 (VLM)
    鈹?                              鈹?
    鈹斺攢鈹€鈻?鎵嬪Э浼拌 鈹€鈹€鈻?3D 鍏抽敭鐐?鈹€鈹€鈹€鈹€鈹?
    鈹?
    鈻?
鏍囨敞杈撳嚭 (Annotation)
    鈹?
    鈹溾攢鈹€鈻?鐗╀綋: [{class, bbox, mask, pose}]
    鈹溾攢鈹€鈻?鎵嬮儴: [{joint_angles, contact_points}]
    鈹溾攢鈹€鈻?鍔ㄤ綔: [{action_type, start, end, params}]
    鈹斺攢鈹€鈻?鍦烘櫙: [{scene_type, affordances}]
    鈹?
    鈻?
鏁版嵁闆?(RLDS / LeRobot Format)
    鈹?
    鈹溾攢鈹€鈻?璁粌闆?(80%)
    鈹溾攢鈹€鈻?楠岃瘉闆?(10%)
    鈹斺攢鈹€鈻?娴嬭瘯闆?(10%)
    鈹?
    鈻?
VLA 璁粌 (OpenVLA / Octo)
    鈹?
    鈻?
妯″瀷妫€鏌ョ偣 鈹€鈹€鈻?TensorRT 浼樺寲 鈹€鈹€鈻?Jetson 閮ㄧ讲
```

### 2.5 寤惰繜涓庢€ц兘棰勭畻

| 闃舵 | 寤惰繜棰勭畻 | 瀹為檯鐩爣 | 浼樺寲绛栫暐 |
|------|----------|----------|----------|
| 閲囬泦-鍚屾 | < 5ms | 2ms | 纭欢瑙﹀彂 + 闆舵嫹璐?|
| 娣卞害浼拌 | < 50ms | 30ms | TensorRT + 鍗婄簿搴?|
| 鎵嬪Э浼拌 | < 30ms | 20ms | 妯″瀷閲忓寲 |
| SLAM 璺熻釜 | < 20ms | 15ms | 鍏抽敭甯х瓥鐣?|
| VLA 鎺ㄧ悊 | < 200ms | 150ms | KV-cache + 鎵瑰鐞?|
| 鍔ㄤ綔鎵ц | < 100ms | 80ms | 杞ㄨ抗棰勮绠?|
| **绔埌绔?* | **< 500ms** | **400ms** | 娴佹按绾垮苟琛?|

### 2.6 瀹归敊璁捐

| 鏁呴殰鍦烘櫙 | 妫€娴嬫柟寮?| 鎭㈠绛栫暐 |
|----------|----------|----------|
| 鐩告満涓㈠抚 | 甯у彿杩炵画鎬ф鏌?| 鎻掑€艰ˉ甯?/ 鍛婅 |
| IMU 楗卞拰 | 鏁板€艰竟鐣屾鏌?| 闄嶇骇涓虹函瑙嗚 SLAM |
| SLAM 涓㈠け | 璺熻釜鐘舵€佺洃鎺?| 閲嶅畾浣?/ 鍥為€€鍒板凡鐭ヤ綅濮?|
| 娣卞害浼拌澶辫触 | 缃俊搴﹂槇鍊?| 浣跨敤涓婁竴甯ф繁搴?+ 鍛婅 |
| VLA 浣庣疆淇″害 | 杈撳嚭鐔垫鏌?| 璇锋眰浜哄伐纭 / 瀹夊叏濮挎€?|
| 缃戠粶涓柇 | 蹇冭烦妫€娴?| 鏈湴缂撳瓨 + 鏂偣缁紶 |

---

## 搂3 鈥?Case 2: Sim2Real 璁粌娴佹按绾?

### 3.1 鍦烘櫙鎻忚堪

鍦ㄤ豢鐪熺幆澧冧腑澶ц妯¤缁冩搷浣滅瓥鐣ワ紝閫氳繃鍩熼殢鏈哄寲鍜屽煙閫傚簲鎶€鏈紝灏嗙瓥鐣ヨ縼绉诲埌鐪熷疄鏈哄櫒浜轰笂鎵ц銆傞€傜敤浜庢暟鎹█缂烘垨鍗遍櫓鍦烘櫙鐨勬搷浣滃涔犮€?

### 3.2 鏋舵瀯鍥?

```mermaid
graph TB
    subgraph "Simulation Layer"
        SIM[浠跨湡鍣?br/>Isaac Sim / MuJoCo]
        DR[鍩熼殢鏈哄寲<br/>绾圭悊/鍏夌収/鐗╃悊]
        SCENE[鍦烘櫙鐢熸垚<br/>绋嬪簭鍖栫敓鎴怾
        ROBOT[鏈哄櫒浜烘ā鍨?br/>URDF/MJCF]
        SIM --> DR
        SCENE --> SIM
        ROBOT --> SIM
    end

    subgraph "Training Layer"
        POLICY[绛栫暐璁粌<br/>PPO / SAC / BC]
        DATA[浠跨湡鏁版嵁闆?br/>鐧句竾绾ц建杩筣
        CURRICULUM[璇剧▼瀛︿範<br/>闅惧害娓愯繘]
        POLICY --> DATA
        CURRICULUM --> POLICY
    end

    subgraph "Adaptation Layer"
        GAP[鍩熷樊璺濆垎鏋?br/>鐗瑰緛鍒嗗竷瀵规瘮]
        ADA[鍩熼€傚簲<br/>DANN / GRDA]
        FINETUNE[鐪熷疄鏁版嵁寰皟<br/>鍗冪骇杞ㄨ抗]
        GAP --> ADA --> FINETUNE
    end

    subgraph "Real World Layer"
        REAL[鐪熷疄鏈哄櫒浜?br/>Franka / UR5]
        CAM[鐪熷疄鐩告満<br/>鏍″噯瀵归綈]
        DEPLOY[閮ㄧ讲鎵ц<br/>瀹夊叏鐩戞帶]
        CAM --> REAL
        ADA --> DEPLOY
    end

    DR --> POLICY
    DATA --> GAP
    FINETUNE --> DEPLOY
```

### 3.3 缁勪欢娓呭崟

| 缁勪欢 | 閫夊瀷 | 鑱岃矗 | 鏇夸唬鏂规 |
|------|------|------|----------|
| 鐗╃悊浠跨湡鍣?| NVIDIA Isaac Sim | 楂樹繚鐪?GPU 鍔犻€熶豢鐪?| MuJoCo, PyBullet, Drake |
| 鍩熼殢鏈哄寲 | Isaac Sim Domain Randomization | 瑙嗚+鐗╃悊鍙傛暟闅忔満 | 鑷畾涔夋彃浠?|
| 鍦烘櫙鐢熸垚 | ProcTHOR / AI2-THOR | 绋嬪簭鍖栧鍐呭満鏅?| BlenderProc, Infinigen |
| 绛栫暐璁粌 | Stable-Baselines3 / RLlib | RL 绠楁硶瀹炵幇 | CleanRL, Tianshou |
| 妯′豢瀛︿範 | Robomimic / LeRobot | BC / Imitation Learning | Diffusion Policy |
| 鍩熼€傚簲 | DANN / AdaBN | 鐗瑰緛绾у煙閫傚簲 | CycleGAN (鍍忕礌绾? |
| 鐪熷疄鏈哄櫒浜?| Franka Emika Panda | 7-DoF 鍗忎綔鑷?| UR5, xArm |
| 瑙嗚鎰熺煡 | 浠跨湡-鐪熷疄鍏变韩缂栫爜鍣?| ResNet/ViT 楠ㄥ共 | Domain Randomization 棰勮缁?|

### 3.4 鏁版嵁娴?

```
浠跨湡鐜閰嶇疆
    鈹?
    鈹溾攢鈹€鈻?鍦烘櫙鍙傛暟 (鎴块棿甯冨眬銆佸鍏枫€佸厜鐓?
    鈹溾攢鈹€鈻?鐗╀綋鍙傛暟 (褰㈢姸銆佹潗璐ㄣ€佽川閲忋€佹懇鎿?
    鈹溾攢鈹€鈻?鏈哄櫒浜哄弬鏁?(鍏宠妭闄愪綅銆侀€熷害銆佸姞閫熷害)
    鈹斺攢鈹€鈻?鐩告満鍙傛暟 (浣嶅Э銆丗OV銆佸櫔澹版ā鍨?
    鈹?
    鈻?
鍩熼殢鏈哄寲 (姣?episode 闅忔満閲囨牱)
    鈹?
    鈹溾攢鈹€鈻?瑙嗚闅忔満: 绾圭悊銆侀鑹层€佸厜鐓с€佽儗鏅?
    鈹溾攢鈹€鈻?鐗╃悊闅忔満: 璐ㄩ噺銆佹懇鎿︺€佹仮澶嶇郴鏁?
    鈹溾攢鈹€鈻?鍔ㄥ姏瀛﹂殢鏈? 鍏宠妭闃诲凹銆佹墽琛屽櫒澧炵泭
    鈹斺攢鈹€鈻?瑙傛祴闅忔満: 鐩告満鍣０銆佸欢杩熴€?dropout
    鈹?
    鈻?
绛栫暐浜や簰閲囬泦
    鈹?
    鈹溾攢鈹€鈻?鐘舵€? {rgb, depth, proprioception}
    鈹溾攢鈹€鈻?鍔ㄤ綔: {joint_pos, joint_vel, gripper}
    鈹溾攢鈹€鈻?濂栧姳: {task_success, efficiency, safety}
    鈹斺攢鈹€鈻?杞ㄨ抗: [(s_0, a_0, r_0), ..., (s_T, a_T, r_T)]
    鈹?
    鈻?
澶ц妯′豢鐪熸暟鎹泦 (1M+ trajectories)
    鈹?
    鈹溾攢鈹€鈻?棰勮缁? 閫氱敤鎿嶄綔鎶€鑳?(鎶撳彇銆佹斁缃€佹帹鍔?
    鈹溾攢鈹€鈻?璇剧▼瀛︿範: 闅惧害娓愯繘 (鍗曠墿浣?鈫?澶氱墿浣?鈫?閬尅)
    鈹斺攢鈹€鈻?绛栫暐妫€鏌ョ偣: 姣?N 姝ヤ繚瀛?
    鈹?
    鈻?
鍩熷樊璺濆垎鏋?
    鈹?
    鈹溾攢鈹€鈻?鐗瑰緛鍒嗗竷: 浠跨湡 vs 鐪熷疄 CNN 鐗瑰緛鐩存柟鍥?
    鈹溾攢鈹€鈻?鎬ц兘宸窛: 浠跨湡鎴愬姛鐜?vs 鐪熷疄闆舵牱鏈垚鍔熺巼
    鈹斺攢鈹€鈻?鍙鍖栧樊璺? Grad-CAM 娉ㄦ剰鍔涘浘瀵规瘮
    鈹?
    鈻?
鍩熼€傚簲绛栫暐
    鈹?
    鈹溾攢鈹€鈻?DANN: 瀵规姉璁粌鍩熷垎绫诲櫒
    鈹溾攢鈹€鈻?AdaBN: 鎵归噺褰掍竴鍖栫粺璁￠噺閫傞厤
    鈹溾攢鈹€鈻?CORAL: 浜岄樁缁熻閲忓榻?
    鈹斺攢鈹€鈻?鐪熷疄鏁版嵁寰皟: 鍗冪骇鐪熷疄杞ㄨ抗绮捐皟
    鈹?
    鈻?
鐪熷疄閮ㄧ讲
    鈹?
    鈹溾攢鈹€鈻?鎰熺煡缂栫爜鍣?(浠跨湡棰勮缁冩潈閲?
    鈹溾攢鈹€鈻?绛栫暐缃戠粶 (鍩熼€傚簲鍚?
    鈹溾攢鈹€鈻?瀹夊叏灞? 纰版挒妫€娴嬨€侀€熷害闄愬埗銆佺揣鎬ュ仠姝?
    鈹斺攢鈹€鈻?鎵ц: 鐪熷疄鏈哄櫒浜哄叧鑺傛帶鍒?
```

### 3.5 Sim2Real Gap 閲忓寲

| Gap 绫诲瀷 | 搴﹂噺鏂瑰紡 | 鍏稿瀷鍊?| 缂╁皬鏂规硶 |
|----------|----------|--------|----------|
| 瑙嗚 Gap | FID (Fr茅chet Inception Distance) | 50-150 | 鍩熼殢鏈哄寲 + 椋庢牸杩佺Щ |
| 鍔ㄥ姏瀛?Gap | 杞ㄨ抗璺熻釜 RMSE | 0.1-0.3 rad | 绯荤粺杈ㄨ瘑 + 闅忔満鍖?|
| 鎺ヨЕ Gap | 纰版挒妫€娴嬫椂搴忓樊 | 10-50ms | 鎺ヨЕ妯″瀷缁嗗寲 |
| 瑙傛祴 Gap | 鐗瑰緛绌洪棿 MMD | 0.5-2.0 | DANN / 鍏变韩缂栫爜鍣?|

### 3.6 鍙墿灞曟€ц璁?

```
浠跨湡骞惰鍖?
    鈹?
    鈹溾攢鈹€鈻?鍗曟満澶氳繘绋? 16 envs 脳 1 GPU
    鈹溾攢鈹€鈻?澶氭満鍒嗗竷寮? 1000+ envs 脳 64 GPUs
    鈹斺攢鈹€鈻?浜戝脊鎬т几缂? Kubernetes + GPU 鑺傜偣姹?
    鈹?
    鈻?
璁粌鎵╁睍
    鈹?
    鈹溾攢鈹€鈻?鏁版嵁骞惰: 澶?GPU 鍚屾姊害
    鈹溾攢鈹€鈻?妯″瀷骞惰: 澶х瓥鐣ョ綉缁滃垎鐗?
    鈹斺攢鈹€鈻?寮傛璁粌: IMPALA / Ape-X 椋庢牸
    鈹?
    鈻?
璇勪及鎵╁睍
    鈹?
    鈹溾攢鈹€鈻?鑷姩鍖栬瘎浼? 棰勫畾涔夋祴璇曞満鏅泦
    鈹溾攢鈹€鈻?鎸囨爣鑱氬悎: 璺ㄧ幆澧冩垚鍔熺巼缁熻
    鈹斺攢鈹€鈻?鍥炲綊妫€娴? 鏂扮増鏈?vs 鍩虹嚎瀵规瘮
```

---

## 搂4 鈥?Case 3: 閬ユ搷浣滄暟鎹敹闆嗕笌 VLA 璁粌

### 4.1 鍦烘櫙鎻忚堪

閫氳繃浜虹被涓撳閬ユ搷浣滄満鍣ㄤ汉鎵ц澶嶆潅浠诲姟锛岄噰闆嗛珮璐ㄩ噺婕旂ず鏁版嵁锛岀敤浜庤缁?VLA 妯″瀷瀹炵幇鑷富鎿嶄綔銆傞€傜敤浜庨渶瑕佺簿缁嗘搷浣滄妧鑳界殑鍙岃噦鍗忎綔鍦烘櫙銆?

### 4.2 鏋舵瀯鍥?

```mermaid
graph TB
    subgraph "Teleoperation Layer"
        OP[鎿嶄綔鍛榏
        VR[VR 澶存樉<br/>Quest 3 / Vision Pro]
        CTRL[鎵嬫焺/鎵嬪<br/>鎺у埗鍣╙
        MASTER[涓昏噦<br/>鍔涘弽棣堣澶嘳
        OP --> VR
        OP --> CTRL
        CTRL --> MASTER
    end

    subgraph "Robot Layer"
        SLAVE[浠庤噦<br/>鍙岃噦鏈哄櫒浜篯
        CAM1[澶撮儴鐩告満<br/>鍏ㄥ眬瑙嗚]
        CAM2[鑵曢儴鐩告満<br/>灞€閮ㄨ瑙抅
        CAM3[绗笁瑙嗚<br/>鍥哄畾鏈轰綅]
        FT[鍔?鍔涚煩浼犳劅鍣╙
        SLAVE --> FT
    end

    subgraph "Synchronization Layer"
        SYNC[鏃堕棿鍚屾鏈嶅姟鍣?br/>PTP Grandmaster]
        BUFFER[鐜舰缂撳啿鍖?br/>10s 婊戝姩绐楀彛]
        RECORD[ROS2 褰曞埗<br/>MCAP 鏍煎紡]
    end

    subgraph "Training Pipeline"
        ALIGN[鍔ㄤ綔瀵归綈<br/>涓讳粠鏄犲皠]
        FILTER[鏁版嵁娓呮礂<br/>寮傚父鍘婚櫎]
        AUG[鏁版嵁澧炲己<br/>鏃跺簭/绌洪棿]
        VLA[VLA 璁粌<br/>鍔ㄤ綔棰勬祴]
    end

    MASTER -. 缃戠粶 .-> SLAVE
    CAM1 --> SYNC
    CAM2 --> SYNC
    CAM3 --> SYNC
    FT --> SYNC
    MASTER --> SYNC
    SYNC --> BUFFER --> RECORD
    RECORD --> ALIGN --> FILTER --> AUG --> VLA
```

### 4.3 缁勪欢娓呭崟

| 缁勪欢 | 閫夊瀷 | 鑱岃矗 | 鏇夸唬鏂规 |
|------|------|------|----------|
| VR 澶存樉 | Meta Quest 3 | 娌夋蹈寮忚瑙夊弽棣?| Apple Vision Pro, HTC Vive |
| 涓昏噦 | Force Dimension Sigma.7 | 7-DoF 鍔涘弽棣?| Geomagic Touch, haption |
| 浠庤噦 | Franka Emika Panda 脳 2 | 鍙岃噦鍗忎綔鎿嶄綔 | UR5e, xArm7 |
| 鍏ㄥ眬鐩告満 | Azure Kinect DK | 鍦烘櫙 RGB-D | RealSense D455, ZED 2i |
| 鑵曢儴鐩告満 | RealSense D405 | 鎵?鐗╀氦浜掔壒鍐?| 寰瀷 USB 鐩告満 |
| 鍔涗紶鎰熷櫒 | ATI Nano43 | 鍏淮鍔?鍔涚煩 | Robotiq, OnRobot |
| 鏃堕棿鍚屾 | PTP Grandmaster (LinuxPTP) | 浜氭绉掑悓姝?| NTP + 杞欢鍚屾 |
| 褰曞埗鏍煎紡 | MCAP | 澶氳瘽棰橀珮鏁堝瓨鍌?| ROS bag v2, HDF5 |
| 鍔ㄤ綔鏄犲皠 | 绗涘崱灏旂┖闂存槧灏?| 涓讳粠浣嶇疆鏄犲皠 | 鍏宠妭绌洪棿鏄犲皠 |

### 4.4 鏁版嵁娴?

```
閬ユ搷浣滀細璇?(Teleop Session)
    鈹?
    鈹溾攢鈹€鈻?鎿嶄綔鍛橀€氳繃 VR 瑙傚療浠庤噦鐜
    鈹溾攢鈹€鈻?鎿嶄綔鍛樼Щ鍔ㄤ富鑷傦紝鍔涘弽棣堜紶閫掕Е鎰?
    鈹溾攢鈹€鈻?涓昏噦浣嶅Э瀹炴椂鏄犲皠鍒颁粠鑷傜洰鏍囦綅濮?
    鈹溾攢鈹€鈻?浠庤噦鎵ц鍔ㄤ綔锛屽姏浼犳劅鍣ㄥ弽棣堟帴瑙︿俊鎭?
    鈹斺攢鈹€鈻?澶氳瑙掕棰?+ 浼犳劅鍣ㄦ暟鎹悓姝ュ綍鍒?
    鈹?
    鈻?
鍚屾鏁版嵁娴?(1 kHz 鎺у埗, 30-60 Hz 瑙嗚)
    鈹?
    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
    鈹?Topic                          Rate    Description      鈹?
    鈹?/master/arm_pose              1 kHz   涓昏噦鏈浣嶅Э      鈹?
    鈹?/slave/left/joint_states      1 kHz   宸﹁噦鍏宠妭鐘舵€?     鈹?
    鈹?/slave/right/joint_states     1 kHz   鍙宠噦鍏宠妭鐘舵€?     鈹?
    鈹?/slave/left/ft_sensor         1 kHz   宸﹁厱鍔?鍔涚煩      鈹?
    鈹?/slave/right/ft_sensor        1 kHz   鍙宠厱鍔?鍔涚煩      鈹?
    鈹?/cameras/head/color           30 Hz   澶撮儴 RGB        鈹?
    鈹?/cameras/head/depth           30 Hz   澶撮儴娣卞害          鈹?
    鈹?/cameras/wrist_left/color     60 Hz   宸﹁厱 RGB          鈹?
    鈹?/cameras/wrist_right/color    60 Hz   鍙宠厱 RGB          鈹?
    鈹?/cameras/third_person/color   30 Hz   绗笁瑙嗚 RGB      鈹?
    鈹?/vr/headset_pose              90 Hz   VR 澶撮儴浣嶅Э       鈹?
    鈹?/vr/controller_poses          90 Hz   鎵嬫焺浣嶅Э          鈹?
    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
    鈹?
    鈻?
鍔ㄤ綔瀵归綈涓庨噸閲囨牱
    鈹?
    鈹溾攢鈹€鈻?涓讳粠鏄犲皠: 绗涘崱灏斾綅缃瘮渚嬫槧灏?+ 濮挎€佺洿鎺ユ槧灏?
    鈹溾攢鈹€鈻?閲嶉噰鏍? 1 kHz 鈫?鎺у埗棰戠巼 (閫氬父 10-50 Hz)
    鈹溾攢鈹€鈻?骞虫粦: 浣庨€氭护娉㈠幓闄ら珮棰戞姈鍔?
    鈹斺攢鈹€鈻?鍔ㄤ綔绌洪棿缁熶竴: 杞崲涓烘ā鍨嬭緭鍏ユ牸寮?(delta pose / joint pos)
    鈹?
    鈻?
鏁版嵁娓呮礂
    鈹?
    鈹溾攢鈹€鈻?鍘婚櫎闈欐娈? 閫熷害 < threshold 瑙嗕负鏃犳晥
    鈹溾攢鈹€鈻?鍘婚櫎寮傚父娈? 鍔犻€熷害瓒呭嚭鐗╃悊鍙兘鑼冨洿
    鈹溾攢鈹€鈻?鍘婚櫎澶辫触娈? 浠诲姟鏈垚鍔熷畬鎴愮殑杞ㄨ抗
    鈹溾攢鈹€鈻?鏃堕棿瀵归綈鏍￠獙: 澶氭ā鎬佹椂闂存埑涓€鑷存€ф鏌?
    鈹斺攢鈹€鈻?璐ㄩ噺璇勫垎: 骞虫粦搴︺€佹晥鐜囥€佹垚鍔熺巼
    鈹?
    鈻?
鏁版嵁澧炲己 (鍙€?
    鈹?
    鈹溾攢鈹€鈻?鏃跺簭澧炲己: 閫熷害缂╂斁 (0.8x - 1.2x)
    鈹溾攢鈹€鈻?绌洪棿澧炲己: 鐩爣浣嶇疆鎵板姩銆佺浉鏈轰綅濮挎壈鍔?
    鈹溾攢鈹€鈻?瑙嗚澧炲己: 棰滆壊鎶栧姩銆佸厜鐓у彉鍖栥€佽儗鏅浛鎹?
    鈹斺攢鈹€鈻?鍔ㄤ綔澧炲己: 璧峰浣嶅Э鎵板姩銆佽矾寰勬彃鍊?
    鈹?
    鈻?
VLA 璁粌鏁版嵁鏍煎紡
    鈹?
    鈹溾攢鈹€鈻?瑙嗚杈撳叆: [T, H, W, 3] 瑙嗛甯у簭鍒?
    鈹溾攢鈹€鈻?璇█鎸囦护: "鐢ㄥ乏鎵嬫嬁璧锋澂瀛愶紝閫掔粰鍙虫墜"
    鈹溾攢鈹€鈻?鍔ㄤ綔杈撳嚭: [T, action_dim] 鐩爣鍏宠妭浣嶇疆 / 鏈浣嶅Э
    鈹溾攢鈹€鈻?杈呭姪杈撳叆: 鍔?鍔涚煩淇℃伅 (鍙€?
    鈹斺攢鈹€鈻?鍏冩暟鎹? {task_id, success, duration, operator_id}
    鈹?
    鈻?
VLA 璁粌 (OpenVLA / Octo / Diffusion Policy)
    鈹?
    鈹溾攢鈹€鈻?棰勮缁? 澶ц妯￠仴鎿嶄綔鏁版嵁闆?(Open X-Embodiment)
    鈹溾攢鈹€鈻?寰皟: 鐗瑰畾浠诲姟/鏈哄櫒浜哄钩鍙版暟鎹?
    鈹斺攢鈹€鈻?璇勪及: 鐪熷疄鐜鑷富鎵ц鎴愬姛鐜?
```

### 4.5 寤惰繜鍒嗘瀽

| 閾捐矾 | 寤惰繜鏉ユ簮 | 鍏稿瀷鍊?| 浼樺寲鐩爣 |
|------|----------|--------|----------|
| 涓昏噦閲囨牱 | USB/浠ュお缃戦€氫俊 | 1-2ms | < 1ms |
| 缃戠粶浼犺緭 | 涓讳粠缃戠粶寤惰繜 | 1-10ms (LAN) | < 5ms |
| 浠庤噦鎺у埗 | 浼烘湇鐜懆鏈?| 1-4ms | < 2ms |
| 瑙嗚鍙嶉 | 鐩告満閲囬泦+缂栫爜+浼犺緭 | 30-50ms | < 33ms |
| VR 娓叉煋 | 鍥惧儚澶勭悊+鏄剧ず | 10-20ms | < 16ms |
| **鎰熺煡鐜欢杩?* | | **43-86ms** | **< 50ms** |
| **鎬绘帶鍒跺欢杩?* | | **5-16ms** | **< 10ms** |

**寤惰繜琛ュ伩绛栫暐**

| 绛栫暐 | 鎻忚堪 | 鏁堟灉 |
|------|------|------|
| 棰勬祴鏄剧ず | 鍩轰簬杩愬姩瀛﹂娴嬫覆鏌撴湭鏉ュ抚 | 鍑忓皯 10-20ms 鎰熺煡寤惰繜 |
| 灞€閮ㄦ覆鏌?| 鍦?VR 绔娴嬫€ф覆鏌?| 鍑忓皯缃戠粶浼犺緭寤惰繜 |
| 鍓嶉鎺у埗 | 鍩轰簬涓昏噦閫熷害棰勬祴浠庤噦鐩爣 | 鍑忓皯鍝嶅簲寤惰繜 |
| 鍔涘弽棣堥娴?| 鍩轰簬鎺ヨЕ妯″瀷棰勬祴鍔涗俊鍙?| 鎻愬崌鎿嶄綔娌夋蹈鎰?|

### 4.6 鏁版嵁璐ㄩ噺淇濊瘉

| 妫€鏌ラ」 | 鏂规硶 | 闃堝€?|
|--------|------|------|
| 涓讳粠鍚屾璇樊 | 鏃堕棿鎴冲樊鍒?| < 2ms |
| 鍔涗紶鎰熷櫒楗卞拰 | 鏁板€艰寖鍥存鏌?| < 95% 閲忕▼ |
| 杩愬姩骞虫粦搴?| 鍔犲姞閫熷害 (jerk) 妫€鏌?| < 鐗╃悊闄愬€?|
| 浠诲姟鎴愬姛鐜?| 浜哄伐鏍囨敞缁撴灉 | > 90% |
| 杞ㄨ抗澶氭牱鎬?| 宓屽叆绌洪棿瑕嗙洊搴?| 鏃犺仛绫婚泦涓?|
| 鎿嶄綔鍛樹竴鑷存€?| 鍚屼竴浠诲姟澶氭搷浣滃憳 Kappa | > 0.75 |

---

## 搂5 鈥?璺ㄦ渚嬪姣?

### 5.1 鏋舵瀯鐗瑰緛瀵规瘮

| 缁村害 | Case 1: Ego-centric | Case 2: Sim2Real | Case 3: Teleoperation |
|------|---------------------|------------------|----------------------|
| **鏁版嵁瑙勬ā** | 涓瓑 (10K-100K 杞ㄨ抗) | 鏋佸ぇ (1M+ 浠跨湡杞ㄨ抗) | 灏忎絾绮?(1K-10K 杞ㄨ抗) |
| **鏁版嵁璐ㄩ噺** | 涓?(鑷姩鏍囨敞涓轰富) | 浣?(浠跨湡绠€鍖? | 楂?(涓撳婕旂ず) |
| **鍩熷樊璺?* | 灏?(鐪熷疄鍒扮湡瀹? | 澶?(浠跨湡鍒扮湡瀹? | 鏃?(鐪熷疄閲囬泦) |
| **鏍囨敞鎴愭湰** | 涓?(VLM + 浜哄伐鎶芥) | 浣?(鑷姩濂栧姳) | 楂?(涓撳鏃堕棿) |
| **璁粌鎴愭湰** | 涓?| 楂?(闇€澶ц妯?RL) | 浣?(BC 涓轰富) |
| **閮ㄧ讲澶嶆潅搴?* | 涓?(杈圭紭浼樺寲) | 楂?(鍩熼€傚簲) | 浣?(鐩存帴杩佺Щ) |
| **娉涘寲鑳藉姏** | 涓?| 楂?(浠跨湡澶氭牱鎬? | 浣?(浠诲姟鐗瑰畾) |
| **瀹夊叏鎬?* | 涓?(鑷富鎵ц) | 楂?(浠跨湡楠岃瘉) | 楂?(浜哄湪鍥炶矾) |

### 5.2 鎶€鏈€夊瀷鐭╅樀

| 鎶€鏈鍩?| Ego-centric 鎺ㄨ崘 | Sim2Real 鎺ㄨ崘 | 閬ユ搷浣滄帹鑽?|
|----------|-----------------|---------------|----------|
| 浠跨湡鍣?| - | Isaac Sim / MuJoCo | - |
| 娣卞害浼拌 | FoundationStereo | 浠跨湡鐩存帴杈撳嚭 | FoundationStereo |
| SLAM | ORB-SLAM3 | - | 鍙€?(鍦烘櫙閲嶅缓) |
| 鎵嬪Э浼拌 | HaMeR | 浠跨湡鍏宠妭鐩存帴璇诲彇 | HaMeR |
| 鏍囨敞鏂瑰紡 | VLM 鑷姩 + 浜哄伐鏍￠獙 | 鑷姩濂栧姳 | 鍔ㄤ綔鐩存帴璁板綍 |
| 璁粌绠楁硶 | VLA (OpenVLA) | RL (PPO/SAC) + BC | BC / Diffusion Policy |
| 鍩熼€傚簲 | 杞婚噺 (鍙€? | 閲嶅害 (蹇呴』) | 涓嶉渶瑕?|
| 閮ㄧ讲骞冲彴 | Jetson / 杈圭紭 | 浜戠 + 杈圭紭 | 宸ヤ綔绔?+ 鏈哄櫒浜?|

## 搂6 鈥?鍏辨€ц璁″師鍒?

### 6.1 寤惰繜浼樺寲

涓変釜妗堜緥鍏变韩鐨勫欢杩熶紭鍖栫瓥鐣ワ細

| 绛栫暐 | 鎻忚堪 | 閫傜敤鍦烘櫙 |
|------|------|----------|
| 娴佹按绾垮苟琛?| 鎰熺煡銆佽鍒掋€佹帶鍒跺苟琛屾墽琛?| 鎵€鏈夋渚?|
| 妯″瀷閲忓寲 | INT8 / FP16 鎺ㄧ悊鍔犻€?| Ego-centric, 閬ユ搷浣?|
| KV-cache | LLM/VLM 鎺ㄧ悊缂撳瓨 | VLA 鎺ㄧ悊 |
| 寮傛 I/O | 鏁版嵁鍔犺浇涓庤绠楅噸鍙?| 璁粌闃舵 |
| 闆舵嫹璐濅紶杈?| 鍏变韩鍐呭瓨 / GPU Direct | 閲囬泦-鎰熺煡閾捐矾 |

### 6.2 瀹归敊璁捐

| 灞傜骇 | 鏈哄埗 | 瀹炵幇 |
|------|------|------|
| 纭欢 | 浼犳劅鍣ㄥ啑浣欍€佺儹鎻掓嫈 | 澶氱浉鏈恒€佸弻 IMU |
| 杞欢 | 寮傚父鎹曡幏銆佷紭闆呴檷绾?| try-catch + 榛樿琛屼负 |
| 缃戠粶 | 鏂嚎閲嶈繛銆佹湰鍦扮紦瀛?| ROS2 DDS QoS |
| 妯″瀷 | 缃俊搴﹂槇鍊笺€佸畨鍏ㄧ瓥鐣?| 浣庣疆淇″害鈫掍繚瀹堝姩浣?|
| 绯荤粺 | 鐩戞帶鍛婅銆佷汉宸ヤ粙鍏?| 瀹炴椂浠〃鏉?|

### 6.3 鍙墿灞曟€?

```
姘村钩鎵╁睍缁村害:
    鈹溾攢鈹€鈻?鏁版嵁骞惰: 澶氭満鍣ㄤ汉鍚屾椂閲囬泦
    鈹溾攢鈹€鈻?浠跨湡骞惰: 澶氱幆澧冨悓鏃惰缁?
    鈹溾攢鈹€鈻?妯″瀷骞惰: 澶?GPU 鍒嗗竷寮忚缁?
    鈹斺攢鈹€鈻?鎺ㄧ悊骞惰: 鎵归噺澶勭悊 + 妯″瀷鍒嗙墖

鍨傜洿鎵╁睍缁村害:
    鈹溾攢鈹€鈻?浼犳劅鍣ㄦ墿灞? 澧炲姞妯℃€?(瑙﹁銆佸惉瑙?
    鈹溾攢鈹€鈻?浠诲姟鎵╁睍: 鏂版搷浣滅被鍒閲忓涔?
    鈹溾攢鈹€鈻?骞冲彴鎵╁睍: 鏂版満鍣ㄤ汉骞冲彴閫傞厤
    鈹斺攢鈹€鈻?鍦烘櫙鎵╁睍: 鏂扮幆澧冨煙閫傚簲
```

## 搂7 鈥?DVAS 椤圭洰鍏宠仈

涓変釜妗堜緥瑕嗙洊浜?DVAS 妗嗘灦鐨勬牳蹇冨簲鐢ㄥ満鏅細

| DVAS 鑳藉姏 | Case 1 瀹炵幇 | Case 2 瀹炵幇 | Case 3 瀹炵幇 |
|-----------|-------------|-------------|-------------|
| 鏁版嵁鐗堟湰绠＄悊 | DVC + 璇箟鐗堟湰 | 浠跨湡鍙傛暟鐗堟湰鍖?| 浼氳瘽绾х増鏈?|
| 璐ㄩ噺闂ㄦ帶 | 鍥涚淮搴︽娴?| 浠跨湡濂栧姳闃堝€?| 涓撳涓€鑷存€?|
| 琛€缂樿拷韪?| 瀹屾暣 pipeline 璁板綍 | 鍙傛暟-绛栫暐鏄犲皠 | 鎿嶄綔鍛?杞ㄨ抗鍏宠仈 |
| 澧為噺鏇存柊 | 鏂颁細璇濆閲忓鐞?| 璇剧▼瀛︿範澧為噺 | 鏂颁笓瀹舵暟鎹閲?|
| 澶氭簮鏁村悎 | 澶氫紶鎰熷櫒铻嶅悎 | 澶氫豢鐪熷櫒鑱氬悎 | 澶氳瑙掑悓姝?|

---

*Related: [01-pipeline-patterns](41-pipeline-patterns.md) | [02-quality-gates](42-quality-gates.md) | Prev: [04-data-ecosystem](../INDEX.md)*
