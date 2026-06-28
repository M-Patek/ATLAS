---
id: multi-sensor-fusion
title: "澶氫紶鎰熷櫒铻嶅悎"
status: complete
complexity: high
related:
  - "./01-stereo-imu.md"
  - "./02-depth-estimation.md"
  - "./03-slam.md"
prerequisites:
  - "姒傜巼璁轰笌鐘舵€佷及璁?
  - "鍗″皵鏇兼护娉㈠熀纭€"
  - "浼樺寲鐞嗚"
last_validated: 2026-06-27
---

# 澶氫紶鎰熷櫒铻嶅悎

## 搂0 鈥?One-liner

澶氫紶鎰熷櫒铻嶅悎閫氳繃鑱斿悎澶勭悊瑙嗚銆両MU銆佹繁搴﹀拰 LiDAR 绛夊紓鏋勪紶鎰熷櫒鏁版嵁锛屽疄鐜版瘮鍗曚竴浼犳劅鍣ㄦ洿椴佹銆佹洿绮剧‘鐨勬劅鐭ヤ笌鐘舵€佷及璁°€?

## 搂1 鈥?浼犳劅鍣ㄧ壒鎬у垎鏋?

### 1.1 鍚勪紶鎰熷櫒浼樺姡鍔?

| 浼犳劅鍣?| 娴嬮噺鍐呭 | 棰戠巼 | 浼樺娍 | 鍔ｅ娍 |
|--------|---------|------|------|------|
| **鐩告満 (RGB)** | 澶栬銆佺汗鐞?| 30-120 Hz | 淇℃伅涓板瘜銆佹垚鏈綆 | 鍏夌収鏁忔劅銆佹棤娣卞害 |
| **鍙岀洰鐩告満** | 瑙嗗樊銆佹繁搴?| 30-60 Hz | 琚姩娴嬮噺銆佽寖鍥村彲璋?| 璁＄畻閲忓ぇ銆佺汗鐞唋ess澶辨晥 |
| **IMU** | 瑙掗€熷害銆佸姞閫熷害 | 200-1000 Hz | 楂橀銆佷綆寤惰繜 | 婕傜Щ銆佸櫔澹?|
| **娣卞害鐩告満 (RGBD)** | 娣卞害銆侀鑹?| 30 Hz | 鐩存帴娣卞害娴嬮噺 | 鑼冨洿鍙楅檺銆佸澶栧樊 |
| **LiDAR** | 3D 鐐逛簯 | 10-20 Hz | 楂樼簿搴︺€佽繙璺濈 | 绋€鐤忋€佹槀璐点€佹満姊伴儴浠?|
| **GPS/RTK** | 鍏ㄥ眬浣嶇疆 | 1-10 Hz | 鍏ㄥ眬瀹氫綅 | 瀹ゅ唴澶辨晥銆佺簿搴︽湁闄?|

### 1.2 浼犳劅鍣ㄤ簰琛ユ€?

```
浼犳劅鍣ㄤ簰琛ュ叧绯?

鐩告満 + IMU:
  鐩告満鎻愪緵澶栬淇℃伅 鈫愨啋 IMU 鎻愪緵楂橀杩愬姩
  鐩告満浣庨 鈫愨啋 IMU 楂橀
  鐩告満灏哄害鏈煡 鈫愨啋 IMU 鎻愪緵缁濆灏哄害

娣卞害 + LiDAR:
  娣卞害鐩告満鎻愪緵绋犲瘑娣卞害 鈫愨啋 LiDAR 鎻愪緵绋€鐤忛珮绮惧害娣卞害
  娣卞害鐩告満杩戣窛绂?鈫愨啋 LiDAR 杩滆窛绂?

瑙嗚 + LiDAR:
  瑙嗚鎻愪緵璇箟 鈫愨啋 LiDAR 鎻愪緵鍑犱綍
  瑙嗚绾圭悊涓板瘜 鈫愨啋 LiDAR 绾圭悊less鍖哄煙鏈夋晥
```

---

## 搂2 鈥?瑙嗚+IMU+娣卞害+LiDAR 铻嶅悎绛栫暐

### 2.1 铻嶅悎鏋舵瀯鍒嗙被

#### 2.1.1 鏉捐€﹀悎 (Loosely Coupled)

```
鐩告満 鈹€鈹€鈫?[瑙嗚閲岀▼璁 鈹€鈹€鈹?
                         鈹溾攢鈹€鈫?[鐘舵€佽瀺鍚圿 鈹€鈹€鈫?杈撳嚭
IMU  鈹€鈹€鈫?[IMU 绉垎] 鈹€鈹€鈹€鈹€鈹?
```

**鐗圭偣**锛?
- 鍚勪紶鎰熷櫒鐙珛澶勭悊
- 铻嶅悎灞傛帴鏀跺悇鑷殑浼拌缁撴灉
- 璁＄畻鏁堢巼楂?
- 绮惧害鍙楅檺浜庡悇瀛愮郴缁?

#### 2.1.2 绱ц€﹀悎 (Tightly Coupled)

```
鐩告満 鈹€鈹€鈹?
       鈹溾攢鈹€鈫?[鑱斿悎浼樺寲] 鈹€鈹€鈫?杈撳嚭
IMU  鈹€鈹€鈹?
娣卞害 鈹€鈹€鈹?
LiDAR 鈹€鈹?
```

**鐗圭偣**锛?
- 鍘熷娴嬮噺鏁版嵁鑱斿悎澶勭悊
- 缁熶竴浼樺寲妗嗘灦
- 绮惧害鏇撮珮
- 璁＄畻澶嶆潅搴﹂珮

### 2.2 瑙嗚-IMU 铻嶅悎

#### 2.2.1 铻嶅悎灞傛

| 灞傛 | 鏂规硶 | 鎻忚堪 |
|------|------|------|
| **鍘熷鏁版嵁** | 鐩存帴娉?VIO | 鍍忕礌鐏板害鑱斿悎浼樺寲 (DSO) |
| **鐗瑰緛鐐?* | 鐗瑰緛鐐规硶 VIO | ORB-SLAM3, VINS-Fusion |
| **甯ч棿鍙樻崲** | 鏉捐€﹀悎 | 瑙嗚浣嶅Э + IMU 绉垎 |
| **浣嶅Э缁撴灉** | 绾澗鑰﹀悎 | 鍗″皵鏇兼护娉㈣瀺鍚?|

#### 2.2.2 瑙嗚-IMU 鑱斿悎鏍囧畾

宸插湪 [01-stereo-imu.md](21-stereo-imu.md) 涓缁嗚璁猴紝姝ゅ琛ュ厖铻嶅悎灞傞潰鐨勮€冭檻锛?

$$\mathbf{z} = \begin{bmatrix} \mathbf{z}_{\text{visual}} \\ \mathbf{z}_{\text{imu}} \end{bmatrix} = \begin{bmatrix} h_{\text{visual}}(\mathbf{x}) + \mathbf{v}_{\text{visual}} \\ h_{\text{imu}}(\mathbf{x}) + \mathbf{v}_{\text{imu}} \end{bmatrix}$$

### 2.3 娣卞害-瑙嗚铻嶅悎

#### 2.3.1 娣卞害杈呭姪瑙嗚

**娣卞害鍦ㄨ瑙変换鍔′腑鐨勪綔鐢?*锛?

| 搴旂敤 | 鏂规硶 | 鏁堟灉 |
|------|------|------|
| **鐗瑰緛鍖归厤** | 娣卞害绾︽潫鐨勬瀬绾挎悳绱?| 鍑忓皯鎼滅储鑼冨洿 |
| **灏哄害鎭㈠** | 娣卞害鍒濆鍖?| 鍗曠洰灏哄害宸茬煡 |
| **椴佹鎬?* | 娣卞害涓€鑷存€ф楠?| 鍓旈櫎璇尮閰?|
| **绋犲瘑閲嶅缓** | RGBD-SLAM | 鐩存帴绋犲瘑鍦板浘 |

#### 2.3.2 瑙嗚杈呭姪娣卞害

**瑙嗚鍦ㄦ繁搴︿及璁′腑鐨勪綔鐢?*锛?

| 搴旂敤 | 鏂规硶 | 鏁堟灉 |
|------|------|------|
| **娣卞害琛ュ叏** | 娣卞害鍥?+ 鍥惧儚 鈫?绋犲瘑娣卞害 | 濉厖绌烘礊 |
| **娣卞害鍘诲櫔** | 杈圭紭寮曞婊ゆ尝 | 淇濇寔杈圭紭 |
| **瓒呭垎杈ㄧ巼** | 鍥惧儚寮曞娣卞害涓婇噰鏍?| 鎻愰珮鍒嗚鲸鐜?|

### 2.4 LiDAR-瑙嗚铻嶅悎

#### 2.4.1 澶栧弬鏍囧畾

LiDAR 涓庣浉鏈虹殑澶栧弬鏍囧畾锛?

$$\mathbf{p}_{\text{camera}} = \mathbf{R}_{\text{cam}}^{\text{lidar}} \mathbf{p}_{\text{lidar}} + \mathbf{t}_{\text{cam}}^{\text{lidar}}$$

**鏍囧畾鏂规硶**锛?

| 鏂规硶 | 鍘熺悊 | 绮惧害 |
|------|------|------|
| **妫嬬洏鏍兼硶** | 妫€娴嬭鐐?+ 鐐逛簯骞抽潰 | 涓瓑 |
| **杈圭紭瀵归綈** | 鍥惧儚杈圭紭涓庣偣浜戣竟缂樺榻?| 楂?|
| **鍩轰簬瀛︿範** | 绔埌绔爣瀹氱綉缁?| 楂?|

#### 2.4.2 娣卞害铻嶅悎绛栫暐

```python
def fuse_lidar_visual(lidar_points, image, depth_map, calibration):
    """
    LiDAR 鐐逛簯涓庤瑙夋繁搴﹁瀺鍚?
    
    杈撳叆:
        - lidar_points: (N, 3) LiDAR 鐐逛簯
        - image: (H, W, 3) RGB 鍥惧儚
        - depth_map: (H, W) 娣卞害鍥?
        - calibration: 澶栧弬鏍囧畾
    
    杈撳嚭:
        - fused_depth: (H, W) 铻嶅悎鍚庣殑娣卞害鍥?
    """
    # 1. 灏?LiDAR 鐐规姇褰卞埌鍥惧儚骞抽潰
    projected = project_points(lidar_points, calibration)
    
    # 2. 娣卞害涓€鑷存€ф楠?
    consistent_mask = check_depth_consistency(projected, depth_map)
    
    # 3. 铻嶅悎
    fused_depth = depth_map.copy()
    fused_depth[consistent_mask] = projected[consistent_mask]
    
    # 4. 杈圭紭淇濇寔婊ゆ尝
    fused_depth = edge_preserving_filter(fused_depth, image)
    
    return fused_depth
```

### 2.5 澶氫紶鎰熷櫒铻嶅悎绯荤粺鏋舵瀯

#### 2.5.1 鍏稿瀷鏋舵瀯

```
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?  鐩告満 1     鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?  鐩告満 2     鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹?
                           鈹?
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?  IMU       鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹?
                           鈹?
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?  娣卞害鐩告満   鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹?
                           鈹?
                    鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                    鈹?  LiDAR     鈹?
                    鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹?
                           鈹?
              鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              鈹?                      鈹?
        [棰勫鐞嗗眰]                [棰勫鐞嗗眰]
              鈹?                      鈹?
        [鐗瑰緛鎻愬彇]                [鐗瑰緛鎻愬彇]
              鈹?                      鈹?
              鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                         鈹?
                   [铻嶅悎灞俔
              鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹粹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              鈹?                  鈹?
        [鐘舵€佷及璁           [鍦板浘鏋勫缓]
              鈹?                  鈹?
              鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
                         鈹?
                    [杈撳嚭]
              (浣嶅Э + 鍦板浘)
```

---

## 搂3 鈥?鍗″皵鏇兼护娉?vs 鍥犲瓙鍥句紭鍖?

### 3.1 鍗″皵鏇兼护娉?(KF/EKF)

#### 3.1.1 鏁板褰㈠紡

**棰勬祴姝?*锛?

$$\mathbf{x}_{k|k-1} = \mathbf{F}_k \mathbf{x}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$

$$\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k$$

**鏇存柊姝?*锛?

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k)^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H}_k \mathbf{x}_{k|k-1})$$

$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$$

#### 3.1.2 EKF 鍦?VIO 涓殑搴旂敤

**MSCKF (Multi-State Constraint Kalman Filter)**锛?

```
鐘舵€佸悜閲?
x = [x_IMU, x_C1, x_C2, ..., x_CN]

鍏朵腑:
  x_IMU = [position, velocity, orientation, bias_a, bias_g]
  x_Ci = [position, orientation] of camera i
```

**浼樺娍**锛?
- 璁＄畻鏁堢巼楂橈紙O(N)锛?
- 瀹炴椂鎬уソ
- 鍐呭瓨鍗犵敤灏?

**灞€闄?*锛?
- 绾挎€у寲璇樊
- 涓嶆敮鎸佸洖鐜?
- 鍏ㄥ眬涓€鑷存€у樊

### 3.2 鍥犲瓙鍥句紭鍖?

#### 3.2.1 鍥犲瓙鍥捐〃绀?

```
鍙橀噺鑺傜偣: x_i (浣嶅Э銆佽矾鏍囩偣)
鍥犲瓙鑺傜偣: f_j (娴嬮噺绾︽潫)

        f1 (IMU)     f2 (瑙嗚)
          鈹?             鈹?
    x1 鈹€鈹€鈹€鈹尖攢鈹€鈹€ x2 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€ x3
          鈹?             鈹?
        f3 (鍏堥獙)     f4 (鍥炵幆)
```

#### 3.2.2 浼樺寲鐩爣

$$\min_{\mathbf{X}} \sum_{i} \rho_i \left\| \mathbf{r}_i(\mathbf{X}) \right\|^2_{\mathbf{\Sigma}_i}$$

鍏朵腑锛?
- $\mathbf{r}_i$锛氭畫宸嚱鏁?
- $\mathbf{\Sigma}_i$锛氫俊鎭煩闃?
- $\rho_i$锛氶瞾妫掓牳鍑芥暟

#### 3.2.3 姹傝В鏂规硶

| 鏂规硶 | 鍘熺悊 | 澶嶆潅搴?| 閫傜敤鍦烘櫙 |
|------|------|--------|---------|
| **Gauss-Newton** | 绾挎€ц繎浼?+ 姝ｈ鏂圭▼ | O(N^3) | 灏忚妯?|
| **Levenberg-Marquardt** | GN + 闃诲凹 | O(N^3) | 閫氱敤 |
| **Schur Complement** | 杈圭紭鍖栬矾鏍囩偣 | O(N) | 澶ц妯?BA |
| **iSAM2** | 澧為噺寮?QR | O(N) | 瀹炴椂 SLAM |
| **GTSAM** | 鍥犲瓙鍥惧紩鎿?| O(N) | 閫氱敤 |

### 3.3 鍗″皵鏇兼护娉?vs 鍥犲瓙鍥句紭鍖栧姣?

| 缁村害 | 鍗″皵鏇兼护娉?(EKF) | 鍥犲瓙鍥句紭鍖?|
|------|-----------------|-----------|
| **璁＄畻澶嶆潅搴?* | O(N) | O(N^3) 鈫?O(N)锛堢█鐤忎紭鍖栵級 |
| **绮惧害** | 涓瓑 | 楂?|
| **鍏ㄥ眬涓€鑷存€?* | 宸?| 濂?|
| **鍥炵幆妫€娴?* | 涓嶆敮鎸?| 鏀寔 |
| **瀹炴椂鎬?* | 浼樼 | 鑹ソ |
| **鍐呭瓨鍗犵敤** | 浣?| 楂?|
| **闈炵嚎鎬у鐞?* | 涓€闃剁嚎鎬у寲 | 杩唬浼樺寲 |
| **涓嶇‘瀹氭€т及璁?* | 鍗忔柟宸煩闃?| 杈圭紭鍗忔柟宸?|
| **浠ｈ〃绯荤粺** | MSCKF, OpenVINS | ORB-SLAM3, Kimera |
| **閫傜敤鍦烘櫙** | 璧勬簮鍙楅檺銆佸疄鏃舵€ц姹傞珮 | 绮惧害瑕佹眰楂樸€侀暱鏈熻繍琛?|

### 3.4 娣峰悎绛栫暐

#### 3.4.1 鍓嶇婊ゆ尝 + 鍚庣浼樺寲

```
鍓嶇 (楂橀):
  EKF-based VIO
  鈹溾攢鈹€ 瀹炴椂浣嶅Э浼拌
  鈹斺攢鈹€ 灞€閮ㄤ竴鑷存€?

鍚庣 (浣庨):
  鍥犲瓙鍥句紭鍖?
  鈹溾攢鈹€ 鍥炵幆妫€娴?
  鈹溾攢鈹€ 鍏ㄥ眬 BA
  鈹斺攢鈹€ 鍏ㄥ眬涓€鑷存€?
```

**浠ｈ〃绯荤粺**锛?
- VINS-Fusion锛氬墠绔?EKF + 鍚庣 pose graph
- ORB-SLAM3锛歍racking (灞€閮?BA) + Loop Closing (鍏ㄥ眬 BA)

#### 3.4.2 婊戝姩绐楀彛浼樺寲

```
绐楀彛鍐呯殑鐘舵€佽仈鍚堜紭鍖?

[x_{k-W}, x_{k-W+1}, ..., x_k]

绾︽潫:
  - IMU 棰勭Н鍒嗙害鏉?
  - 瑙嗚閲嶆姇褰辩害鏉?
  - 杈圭紭鍖栧厛楠?
```

**浼樺娍**锛?
- 骞宠　绮惧害涓庢晥鐜?
- 鍥哄畾璁＄畻閲?
- 鏀寔瀹炴椂鎬?

---

## 搂4 鈥?浼犳劅鍣ㄦ椂闂村榻愪笌绌洪棿閰嶅噯

### 4.1 鏃堕棿瀵归綈

#### 4.1.1 鏃堕棿鍚屾闂

| 闂 | 鍘熷洜 | 褰卞搷 |
|------|------|------|
| **鏃堕棿鎴虫姈鍔?* | 鎿嶄綔绯荤粺璋冨害銆乁SB 寤惰繜 | 娴嬮噺涓嶅悓姝?|
| **鏃堕挓婕傜Щ** | 鍚勪紶鎰熷櫒鐙珛鏃堕挓 | 闀挎湡绱Н璇樊 |
| **浼犺緭寤惰繜** | 缃戠粶/USB 浼犺緭 | 鍥哄畾鎴栭殢鏈哄欢杩?|

#### 4.1.2 鏃堕棿瀵归綈鏂规硶

| 鏂规硶 | 绮惧害 | 澶嶆潅搴?| 閫傜敤鍦烘櫙 |
|------|------|--------|---------|
| **纭欢鍚屾** | < 1渭s | 楂?| 楂樼簿搴﹁姹?|
| **PTP (IEEE 1588)** | < 1渭s | 涓?| 缃戠粶璁惧 |
| **杞欢鍚屾** | 1-10ms | 浣?| 閫氱敤鍦烘櫙 |
| **鎻掑€煎榻?* | 鍙栧喅浜庨鐜?| 浣?| 楂橀浼犳劅鍣?|

#### 4.1.3 IMU 鎻掑€肩ず渚?

```python
def interpolate_imu(imu_data, target_time):
    """
    IMU 鏁版嵁鎻掑€煎埌鐩爣鏃堕棿
    
    杈撳叆:
        - imu_data: [(timestamp, accel, gyro), ...]
        - target_time: 鐩爣鏃堕棿鎴?
    
    杈撳嚭:
        - interpolated: (accel, gyro) at target_time
    """
    # 鎵惧埌鐩搁偦鐨勪袱涓?IMU 娴嬮噺
    idx = bisect.bisect_left([d[0] for d in imu_data], target_time)
    
    if idx == 0:
        return imu_data[0][1], imu_data[0][2]
    if idx >= len(imu_data):
        return imu_data[-1][1], imu_data[-1][2]
    
    t1, a1, g1 = imu_data[idx - 1]
    t2, a2, g2 = imu_data[idx]
    
    # 绾挎€ф彃鍊?
    alpha = (target_time - t1) / (t2 - t1)
    accel = a1 + alpha * (a2 - a1)
    gyro = g1 + alpha * (g2 - g1)
    
    return accel, gyro
```

### 4.2 绌洪棿閰嶅噯

#### 4.2.1 澶栧弬鏍囧畾

宸插湪 [01-stereo-imu.md](21-stereo-imu.md) 涓缁嗚璁恒€?

#### 4.2.2 澶氫紶鎰熷櫒绌洪棿鍙樻崲

```
缁熶竴鍧愭爣绯? 涓栫晫鍧愭爣绯?W

浼犳劅鍣ㄥ潗鏍囩郴:
  - 鐩告満: C
  - IMU: B (body)
  - LiDAR: L

鍙樻崲鍏崇郴:
  P_w = T_wc * P_c
  P_w = T_wb * P_b
  P_w = T_wl * P_l

浼犳劅鍣ㄩ棿鍙樻崲:
  T_cb = T_wc^{-1} * T_wb
  T_cl = T_wc^{-1} * T_wl
```

#### 4.2.3 鍦ㄧ嚎澶栧弬浼拌

```python
def online_extrinsic_calibration(sensor_data, initial_estimate):
    """
    鍦ㄧ嚎澶栧弬浼拌
    
    杈撳叆:
        - sensor_data: 澶氫紶鎰熷櫒鍚屾鏁版嵁
        - initial_estimate: 鍒濆澶栧弬浼拌
    
    杈撳嚭:
        - T: 浼樺寲鍚庣殑澶栧弬
    """
    T = initial_estimate
    
    for data in sensor_data:
        # 鎻愬彇鍚勪紶鎰熷櫒瑙傛祴
        visual_obs = extract_visual(data)
        imu_obs = extract_imu(data)
        lidar_obs = extract_lidar(data)
        
        # 鏋勫缓娈嬪樊
        residuals = []
        residuals.append(visual_residual(visual_obs, T))
        residuals.append(imu_residual(imu_obs, T))
        residuals.append(lidar_residual(lidar_obs, T))
        
        # 浼樺寲
        T = optimize(residuals, T)
    
    return T
```

---

## 搂5 鈥?涓?DVAS 椤圭洰鐨勫叧鑱?

### 5.1 鍦?DVAS 涓殑瀹氫綅

```
DVAS 澶氫紶鎰熷櫒铻嶅悎鏋舵瀯:

[浼犳劅鍣ㄥ眰]
    鈹溾攢鈹€ 鍙岀洰鐩告満 鈹€鈹€鈹?
    鈹溾攢鈹€ IMU 鈹€鈹€鈹€鈹€鈹€鈹€鈹?
    鈹溾攢鈹€ 娣卞害鐩告満 鈹€鈹€鈹尖攢鈹€鈫?[铻嶅悎灞俔 鈹€鈹€鈫?[搴旂敤灞俔
    鈹斺攢鈹€ LiDAR 鈹€鈹€鈹€鈹€鈹?

铻嶅悎灞?
    鈹溾攢鈹€ 鏃堕棿瀵归綈
    鈹溾攢鈹€ 绌洪棿閰嶅噯
    鈹溾攢鈹€ 鐘舵€佷及璁?(EKF / 鍥犲瓙鍥?
    鈹斺攢鈹€ 涓嶇‘瀹氭€ч噺鍖?

搴旂敤灞?
    鈹溾攢鈹€ SLAM
    鈹溾攢鈹€ 鎵嬪娍浼拌
    鈹溾攢鈹€ 鍦烘櫙閲嶅缓
    鈹斺攢鈹€ 瀵艰埅瑙勫垝
```

### 5.2 DVAS 鐗瑰畾闇€姹?

| 闇€姹?| 铻嶅悎绛栫暐 | 棰勬湡鏁堟灉 |
|------|---------|---------|
| **楂樼簿搴︿綅濮?* | 绱ц€﹀悎 VIO | < 1% 婕傜Щ |
| **椴佹璺熻釜** | 瑙嗚 + IMU + 娣卞害 | 绾圭悊less鍖哄煙涓嶄涪澶?|
| **璇箟鐞嗚В** | 瑙嗚 + LiDAR | 甯﹁涔夌殑 3D 鍦板浘 |
| **瀹炴椂鎬?* | 鍓嶇 EKF + 鍚庣浼樺寲 | 30 FPS |
| **闀挎椂闂磋繍琛?* | 鍥炵幆妫€娴?+ 鍏ㄥ眬 BA | 鏃犵疮绉紓绉?|

### 5.3 铻嶅悎绯荤粺鎺ュ彛璁捐

```python
class MultiSensorFusion:
    """
    澶氫紶鎰熷櫒铻嶅悎绯荤粺
    """
    def __init__(self, sensors, fusion_method="tight_coupled"):
        """
        鍒濆鍖栬瀺鍚堢郴缁?
        
        鍙傛暟:
            - sensors: 浼犳劅鍣ㄩ厤缃垪琛?
            - fusion_method: "tight_coupled" 鎴?"loosely_coupled"
        """
        self.sensors = sensors
        self.fusion_method = fusion_method
        
        if fusion_method == "tight_coupled":
            self.estimator = TightlyCoupledEstimator()
        else:
            self.estimator = LooselyCoupledEstimator()
    
    def process(self, measurements):
        """
        澶勭悊澶氫紶鎰熷櫒娴嬮噺
        
        杈撳叆:
            - measurements: 鍚屾鐨勫浼犳劅鍣ㄦ祴閲?
                {
                    'camera': (image_left, image_right),
                    'imu': (accel, gyro),
                    'depth': depth_map,
                    'lidar': point_cloud
                }
        
        杈撳嚭:
            - state: 浼拌鐘舵€?
                {
                    'position': (x, y, z),
                    'orientation': (qw, qx, qy, qz),
                    'velocity': (vx, vy, vz),
                    'covariance': 6x6 matrix
                }
        """
        # 鏃堕棿瀵归綈
        aligned = self.time_align(measurements)
        
        # 绌洪棿閰嶅噯
        registered = self.spatial_register(aligned)
        
        # 鐘舵€佷及璁?
        state = self.estimator.update(registered)
        
        return state
    
    def time_align(self, measurements):
        """鏃堕棿瀵归綈"""
        # 缁熶竴鍒板弬鑰冩椂闂存埑
        reference_time = measurements['camera'][0]
        aligned = {}
        
        for sensor_name, data in measurements.items():
            aligned[sensor_name] = self.interpolate(data, reference_time)
        
        return aligned
    
    def spatial_register(self, measurements):
        """绌洪棿閰嶅噯"""
        registered = {}
        
        for sensor_name, data in measurements.items():
            T = self.get_extrinsic(sensor_name)
            registered[sensor_name] = self.transform(data, T)
        
        return registered
```

---

## 搂6 鈥?鍙傝€冧笌璧勬簮

### 6.1 鍏抽敭璁烘枃

1. **MSCKF** - Mourikis & Roumeliotis (2007) - "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation" (ICRA)
2. **iSAM2** - Kaess et al. (2012) - "iSAM2: Incremental Smoothing and Mapping with the Bayes Tree" (IJR)
3. **GTSAM** - Dellaert (2012) - "Factor Graphs and GTSAM: A Hands-On Introduction" (GT Tech Report)
4. **VINS-Fusion** - Qin et al. (2018) - "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator" (IEEE T-RO)
5. **LIO-SAM** - Shan et al. (2020) - "LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping" (IROS)
6. **R2LIVE** - Lin et al. (2021) - "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual Tightly-Coupled State Estimator and Mapping" (IEEE TPAMI)

### 6.2 寮€婧愬疄鐜?

| 椤圭洰 | 閾炬帴 | 璇存槑 |
|------|------|------|
| GTSAM | https://github.com/borglab/gtsam | 鍥犲瓙鍥句紭鍖栧簱 |
| g2o | https://github.com/RainerKuemmerle/g2o | 鍥句紭鍖栧簱 |
| Ceres Solver | http://ceres-solver.org | 闈炵嚎鎬т紭鍖栧簱 |
| VINS-Fusion | https://github.com/HKUST-Aerial-Robotics/VINS-Fusion | 瑙嗚-鎯€ц瀺鍚?|
| LIO-SAM | https://github.com/TixiaoShan/LIO-SAM | LiDAR-鎯€ц瀺鍚?|
| R2LIVE | https://github.com/hku-mars/r2live | 澶氫紶鎰熷櫒铻嶅悎 |

### 6.3 鐩稿叧鏂囨。

- [鍙岀洰+IMU 鍗忓悓鏍囧畾](21-stereo-imu.md) 鈥?澶氫紶鎰熷櫒铻嶅悎鐨勫墠鎻?
- [娣卞害浼拌](22-depth-estimation.md) 鈥?娣卞害浼犳劅鍣ㄧ殑浣跨敤
- [SLAM 绯荤粺](23-slam.md) 鈥?澶氫紶鎰熷櫒铻嶅悎鐨勬牳蹇冨簲鐢?

---

*Layer: 03-perception | Prev: [3D 閲嶅缓](25-3d-reconstruction.md) | Next: [鎰熺煡灞傜储寮昡(../INDEX.md)*
