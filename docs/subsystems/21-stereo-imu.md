---
id: stereo-imu-calibration
title: "鍙岀洰+IMU鍗忓悓鏍囧畾"
status: complete
complexity: high
related:
  - "./02-depth-estimation.md"
  - "./03-slam.md"
  - "./06-sensor-fusion.md"
prerequisites:
  - "鐩告満鍐呭弬鏍囧畾 (Pinhole model, distortion models)"
  - "IMU鍣０妯″瀷 (Allan Variance)"
  - "闈炵嚎鎬т紭鍖?(Levenberg-Marquardt, Ceres Solver)"
last_validated: 2026-06-27
---

# 鍙岀洰+IMU鍗忓悓鏍囧畾

## 搂0 鈥?One-liner

鍙岀洰+IMU鍗忓悓鏍囧畾閫氳繃鑱斿悎浼樺寲鐩告満鍐呭弬銆佺暩鍙樼郴鏁般€佺浉鏈洪棿澶栧弬浠ュ強鐩告満-IMU鏃剁┖鍙樻崲锛屼负瑙嗚-鎯€х郴缁熸彁渚涙绫崇骇绮惧害鐨勫嚑浣曚笌 temporal 绾︽潫銆?

## 搂1 鈥?鏍稿績闂瀹氫箟

### 1.1 鏍囧畾鍙傛暟绌洪棿

瑙嗚-鎯€х郴缁熺殑鏍囧畾娑夊強浠ヤ笅鍙傛暟闆嗗悎锛?

| 鍙傛暟绫诲埆 | 绗﹀彿 | 缁村害 | 璇存槑 |
|---------|------|------|------|
| 鐩告満鍐呭弬 | $K$ | $3 \times 3$ | 鐒﹁窛 $(f_x, f_y)$銆佷富鐐?$(c_x, c_y)$ |
| 鐣稿彉绯绘暟 | $D$ | 4-8缁?| Radial $(k_1,k_2,k_3)$ + Tangential $(p_1,p_2)$ |
| 鐩告満闂村鍙?| $T_{c_1}^{c_2}$ | SE(3) | 鍙岀洰鍩虹嚎銆佹棆杞€佸钩绉?|
| 鐩告満-IMU澶栧弬 | $T_c^b$ | SE(3) | IMU鍒扮浉鏈虹殑绌洪棿鍙樻崲 |
| 鏃堕棿鍋忕Щ | $t_d$ | 1缁?| IMU涓庣浉鏈虹殑鏃堕棿宸?|
| IMU鍐呭弬 | $T^a, T^g$ | $3 \times 3$ | 鍔犻€熷害璁?闄€铻轰华鏍囧畾鐭╅樀 |

### 1.2 鑱斿悎鏍囧畾鐨勬暟瀛︽湰璐?

鏍囧畾鐨勬湰璐ㄦ槸瑙ｅ喅浠ヤ笅鏈€澶т技鐒朵及璁★紙MLE锛夐棶棰橈細

$$\theta^* = \arg\min_{\theta} \sum_{i} \left\| r_{\text{visual},i}(\theta) \right\|^2_{\Sigma_v} + \sum_{j} \left\| r_{\text{imu},j}(\theta) \right\|^2_{\Sigma_i}$$

鍏朵腑 $r_{\text{visual}}$ 涓洪噸鎶曞奖璇樊锛?r_{\text{imu}}$ 涓篒MU棰勭Н鍒嗘畫宸€?

---

## 搂2 鈥?Kalibr 宸ュ叿閾捐瑙?

### 2.1 Kalibr 鏋舵瀯姒傝堪

Kalibr 鏄敱 ETH Zurich 寮€鍙戠殑寮€婧愯瑙?鎯€ф爣瀹氬伐鍏烽摼锛屾敮鎸佷互涓嬫爣瀹氭ā寮忥細

```
kalibr/
鈹溾攢鈹€ kalibr_calibrate_cameras      # 澶氱浉鏈烘爣瀹?
鈹溾攢鈹€ kalibr_calibrate_imu_camera   # 鐩告満-IMU鏍囧畾
鈹溾攢鈹€ kalibr_calibrate_rs           # Rolling Shutter鏍囧畾
鈹斺攢鈹€ kalibr_bagextractor           # 鏁版嵁棰勫鐞?
```

### 2.2 澶氱浉鏈烘爣瀹?(Multi-Camera Calibration)

#### 2.2.1 鏍囧畾鏉夸笌瑙傛祴妯″瀷

Kalibr 鏀寔涓ょ鏍囧畾鏉匡細

| 鏍囧畾鏉跨被鍨?| 浼樼偣 | 缂虹偣 | 閫傜敤鍦烘櫙 |
|-----------|------|------|---------|
| **AprilGrid** | 鑷姩妫€娴嬨€佹棤姝т箟ID銆侀儴鍒嗛伄鎸″彲鎭㈠ | 闇€瑕佹墦鍗扮簿搴﹂珮 | 楂樼簿搴︽爣瀹?|
| **Checkerboard** | 鍒朵綔绠€鍗曘€丱penCV鍏煎 | 瑙掔偣妫€娴嬪鍏夌収鏁忔劅銆両D姝т箟 | 蹇€熼獙璇?|

AprilGrid 鐨勮鐐规娴嬪熀浜?AprilTag 缂栫爜绯荤粺锛屾瘡涓?tag 鍏锋湁鍞竴ID锛屾敮鎸侊細
- 閮ㄥ垎閬尅涓嬬殑 tag 璇嗗埆
- 浜氬儚绱犵骇瑙掔偣瀹氫綅锛堥€氳繃灞€閮ㄧ伆搴︽搴︿紭鍖栵級
- 鑷姩 tag 灏哄鎺ㄦ柇

#### 2.2.2 閲嶆姇褰辫宸ā鍨?

瀵逛簬妫€娴嬪埌鐨勮鐐?$p_{ij}$锛堢 $i$ 涓?tag 鐨勭 $j$ 涓鐐癸級锛屽叾鍦ㄧ浉鏈哄潗鏍囩郴涓嬬殑 3D 鍧愭爣涓猴細

$$P_{ij}^{W} = T_{W}^{C_k} \cdot P_{ij}^{\text{tag}}$$

閲嶆姇褰辫宸細

$$e_{ij} = \pi(K, D, T_{C}^{W} \cdot P_{ij}^{W}) - \hat{p}_{ij}$$

鍏朵腑 $\pi(\cdot)$ 涓烘姇褰卞嚱鏁帮紝鍖呭惈鐣稿彉妯″瀷銆?

#### 2.2.3 鍙岀洰鍩虹嚎浼樺寲

鍙岀洰澶栧弬鏍囧畾鍚屾椂浼樺寲锛?

$$T_{c_1}^{c_2} = \begin{bmatrix} R_{c_1}^{c_2} & t_{c_1}^{c_2} \\ 0 & 1 \end{bmatrix}$$

鍏抽敭绾︽潫锛?
- **鍩虹嚎闀垮害** $b = \|t_{c_1}^{c_2}\|$ 搴旀帴杩戣璁″€硷紙濡?Intel RealSense D435i 鐨勫熀绾跨害 55mm锛?
- **鏋佺嚎瀵归綈**锛氭爣瀹氬悗宸﹀彸鍥惧儚瀵瑰簲鏋佺嚎搴斿叡绾?

### 2.3 鐩告満-IMU鏍囧畾 (Camera-IMU Calibration)

#### 2.3.1 鏍稿績绠楁硶锛氬熀浜?B-spline 鐨勮繛缁椂闂翠紭鍖?

Kalibr 閲囩敤 **B-spline** 鍙傛暟鍖栫浉鏈鸿建杩癸紝灏嗙鏁ｇ殑 IMU 娴嬮噺涓庤繛缁殑瑙嗚杞ㄨ抗鍏宠仈锛?

$$T(t) = \sum_{i=0}^{n} N_{i,k}(t) \cdot P_i$$

鍏朵腑 $N_{i,k}(t)$ 涓?$k$ 闃?B-spline 鍩哄嚱鏁帮紝$P_i$ 涓烘帶鍒剁偣銆?

**浼樺娍**锛?
- 杞ㄨ抗 $C^k$ 杩炵画锛屼究浜庤绠楅珮闃跺鏁帮紙閫熷害銆佸姞閫熷害锛?
- 鑷劧鍦颁笌 IMU 鐨勮繛缁祴閲忔ā鍨嬪尮閰?
- 鏀寔鏃堕棿鍋忕Щ鐨勪紭鍖?

#### 2.3.2 IMU 璇樊妯″瀷

IMU 娴嬮噺妯″瀷锛?

$$\tilde{a}_t = a_t + b_a + R_{WB}^T \cdot g_W + n_a$$
$$\tilde{\omega}_t = \omega_t + b_g + n_g$$

鍏朵腑锛?
- $b_a, b_g$ 涓洪殢鏈烘父璧?bias
- $n_a \sim \mathcal{N}(0, \sigma_a^2), n_g \sim \mathcal{N}(0, \sigma_g^2)$
- $g_W$ 涓洪噸鍔涘悜閲?

#### 2.3.3 鑱斿悎浼樺寲鐩爣鍑芥暟

$$\min_{\theta} \sum_{k} \left\| e_{\text{visual},k} \right\|^2 + \sum_{m} \left\| e_{\text{imu},m} \right\|^2 + \sum_{n} \left\| e_{\text{bias},n} \right\|^2$$

鍏朵腑锛?
- $e_{\text{visual}}$: 閲嶆姇褰辫宸?
- $e_{\text{imu}}$: IMU 娴嬮噺娈嬪樊锛堝姞閫熷害璁°€侀檧铻轰华锛?
- $e_{\text{bias}}$: Bias 闅忔満娓歌蛋绾︽潫

### 2.4 鏃堕棿鍚屾鏍囧畾

#### 2.4.1 纭欢鍚屾 vs 杞欢鍚屾

| 缁村害 | 纭欢鍚屾 (Hardware Sync) | 杞欢鍚屾 (Software Sync) |
|------|------------------------|------------------------|
| **鍘熺悊** | 閫氳繃鐗╃悊瑙﹀彂淇″彿锛堝 GPIO銆丗SYNC锛夊悓姝ユ洕鍏?| 鍩轰簬鏃堕棿鎴冲榻愶紝浼拌鏃堕棿鍋忕Щ |
| **绮惧害** | < 1渭s锛堝井绉掔骇锛?| 1-10ms锛堟绉掔骇锛?|
| **纭欢瑕佹眰** | 闇€瑕佸悓姝ョ嚎銆佽Е鍙戜俊鍙?| 浠呴渶杞欢鏀寔 |
| **鍏稿瀷瀹炵幇** | Intel RealSense 鐨?External Sync銆丮IPI CSI-2 | ROS `message_filters::ApproximateTime` |
| **閫傜敤鍦烘櫙** | 楂橀€熻繍鍔ㄣ€乂IO/SLAM | 浣庨€熷満鏅€佺绾垮鐞?|

#### 2.4.2 Kalibr 鐨勬椂闂村亸绉讳及璁?

Kalibr 灏嗘椂闂村亸绉?$t_d$ 浣滀负浼樺寲鍙橀噺锛?

$$t_{\text{camera}} = t_{\text{imu}} + t_d$$

浼樺寲杩囩▼涓悓鏃朵及璁★細
- 鐩告満涓?IMU 涔嬮棿鐨勫浐瀹氭椂闂村亸绉?
- 鍚勪紶鎰熷櫒鍐呴儴鐨勬椂閽熸紓绉伙紙濡傛灉瀛樺湪锛?

**鍏稿瀷缁撴灉**锛?
- USB 鐩告満 + IMU锛?t_d \approx 5-20$ ms
- MIPI CSI 鐩告満 + 鏉胯浇 IMU锛?t_d < 1$ ms

### 2.5 Kalibr 鏍囧畾娴佺▼

```bash
# 1. 鍑嗗鏍囧畾鏉匡紙AprilGrid锛?
# 2. 褰曞埗鏍囧畾鏁版嵁锛堝厖鍒嗘縺鍔辨墍鏈夎嚜鐢卞害锛?
rosbag record /cam0/image_raw /cam1/image_raw /imu0 -O calibration.bag

# 3. 杩愯澶氱浉鏈烘爣瀹?
kalibr_calibrate_cameras --bag calibration.bag \
  --target target.yaml --cam chain.yaml

# 4. 杩愯鐩告満-IMU鏍囧畾
kalibr_calibrate_imu_camera --bag calibration.bag \
  --target target.yaml --cam chain.yaml --imu imu.yaml
```

**鏍囧畾鏁版嵁褰曞埗瑕佹眰**锛?
- 鍏呭垎婵€鍔辨墍鏈?6 涓嚜鐢卞害
- 鍖呭惈瓒冲鐨勬棆杞紙婵€娲?gyroscope锛?
- 鍖呭惈瓒冲鐨勫钩绉伙紙婵€娲?accelerometer锛?
- 閬垮厤绾棆杞垨绾钩绉昏繍鍔?
- 鎸佺画鏃堕棿锛?0-60 绉?

---

## 搂3 鈥?Basalt 鏍囧畾妗嗘灦

### 3.1 Basalt 绠€浠?

Basalt 鏄敱 TUM 寮€鍙戠殑瑙嗚-鎯€ч噷绋嬭涓庢爣瀹氭鏋讹紝鍏舵牳蹇冪壒鐐癸細
- 鍩轰簬 **Square Root Bundle Adjustment (SRBA)** 鐨勪紭鍖?
- 鏀寔 **鍦ㄧ嚎鏍囧畾**锛圤nline Calibration锛?
- 鏇撮珮鏁堢殑绋€鐤?Schur complement 姹傝В

### 3.2 Basalt vs Kalibr

| 鐗规€?| Kalibr | Basalt |
|------|--------|--------|
| **浼樺寲妗嗘灦** | B-spline 杩炵画鏃堕棿 | 绂绘暎鏃堕棿 + SRBA |
| **鍦ㄧ嚎鏍囧畾** | 涓嶆敮鎸?| 鏀寔 |
| **璁＄畻鏁堢巼** | 绂荤嚎锛岃緝鎱?| 瀹炴椂鎴栬繎瀹炴椂 |
| **Rolling Shutter** | 鏀寔锛堝崟鐙伐鍏凤級 | 鍘熺敓鏀寔 |
| **浠ｇ爜渚濊禆** | 杈冮噸锛圧OS銆丆eres锛?| 鐩稿杞婚噺 |
| **绀惧尯娲昏穬搴?* | 楂?| 涓瓑 |
| **绮惧害** | 楂橈紙鍩哄噯锛?| 楂橈紙鍙瘮杈冿級 |

### 3.3 Basalt 鐨勬牳蹇冨垱鏂?

#### 3.3.1 Square Root Bundle Adjustment

浼犵粺 BA 鐨勬瑙勬柟绋嬶細

$$J^T \Sigma^{-1} J \cdot \delta x = J^T \Sigma^{-1} r$$

SRBA 閫氳繃 QR 鍒嗚В閬垮厤鏄惧紡鏋勯€?$J^T J$锛?

$$J = QR \Rightarrow R^T R \cdot \delta x = J^T r$$

**浼樺娍**锛?
- 鏁板€肩ǔ瀹氭€ф洿濂斤紙鏉′欢鏁版敼鍠勶級
- 鍙埄鐢ㄧ█鐤?QR 鍒嗚В鍔犻€?
- 閫傚悎澧為噺寮忔洿鏂?

#### 3.3.2 鍦ㄧ嚎鏍囧畾娴佺▼

```
杈撳叆: 瑙嗛娴?+ IMU 鏁版嵁
  鈫?
鍒濆鍖? 绮楃暐澶栧弬浼拌锛堝熀浜庢墜鐪兼爣瀹氾級
  鈫?
璺熻釜: 鍚屾椂杩涜 VIO 鍜屾爣瀹氬弬鏁颁紭鍖?
  鈫?
鏀舵暃鍒ゆ柇: 鏍囧畾鍙傛暟鏂瑰樊 < threshold
  鈫?
杈撳嚭: 绮剧‘鏍囧畾鍙傛暟
```

---

## 搂4 鈥?鏃堕棿鍚屾鏂规硶娣卞害鍒嗘瀽

### 4.1 纭欢鍚屾瀹炵幇

#### 4.1.1 Intel RealSense D435i 鐨勫悓姝ユ満鍒?

```
Master Camera (D435i #1)          Slave Camera (D435i #2)
     鈹?                                   鈹?
     鈹溾攢 GPIO Pin 1 (VSYNC) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€>鈹溾攢 GPIO Pin 1
     鈹?                                   鈹?
     鈹? 鍚屾淇″彿: 姣忓抚鏇濆厜寮€濮嬫椂鍙戦€佽剦鍐?
     鈹? 绮惧害: < 100渭s
```

**閰嶇疆姝ラ**锛?
```bash
# 璁剧疆涓讳粠鐩告満
rs-enumerate-devices  # 鏌ョ湅璁惧搴忓垪鍙?
# 閫氳繃 rsync 宸ュ叿閰嶇疆纭欢鍚屾
```

#### 4.1.2 MIPI CSI-2 鐨勫抚鍚屾

MIPI CSI-2 鎺ュ彛鏀寔閫氳繃 **Frame Sync (FS)** 淇″彿瀹炵幇澶氱浉鏈哄悓姝ワ細

- **FS 淇″彿**锛氱敱 ISP 鎴栧閮ㄦ帶鍒跺櫒浜х敓
- **鍚屾绮惧害**锛氬彈闄愪簬 CSI-2 鐨?LP 妯″紡鏃堕挓锛堥€氬父 < 1渭s锛?
- **鍏稿瀷骞冲彴**锛欽etson Xavier銆丷aspberry Pi CM4

### 4.2 杞欢鍚屾鏂规硶

#### 4.2.1 鍩轰簬鏃堕棿鎴崇殑鏈€杩戦偦鍖归厤

```python
# ROS 涓殑 ApproximateTime 鍚屾
def callback(cam_msg, imu_msg):
    # 鑷姩鍖归厤鏃堕棿鎴虫渶鎺ヨ繎鐨勬秷鎭
    pass

ts = message_filters.ApproximateTimeSynchronizer(
    [cam_sub, imu_sub], queue_size=10, slop=0.01  # 10ms 瀹瑰樊
)
```

#### 4.2.2 鍩轰簬 IMU 绉垎鐨勬椂闂村榻?

鍒╃敤 IMU 鐨勯珮棰戠壒鎬э紙閫氬父 200-1000Hz锛夛紝鍦ㄧ浉鏈哄抚涔嬮棿杩涜鎻掑€硷細

$$\omega(t) = \text{lerp}(\omega_{k}, \omega_{k+1}, t)$$

### 4.3 鏃堕棿鍚屾绮惧害瀵规瘮

| 鏂规硶 | 绮惧害 | 澶嶆潅搴?| 閫傜敤骞冲彴 |
|------|------|--------|---------|
| PTP (IEEE 1588) | < 1渭s | 楂?| 宸ヤ笟鐩告満銆佺綉缁滆澶?|
| GPIO Hardware Sync | < 100渭s | 涓?| RealSense銆佸伐涓氱浉鏈?|
| MIPI FS | < 1渭s | 涓?| Jetson銆佸祵鍏ュ紡骞冲彴 |
| USB UVC Timestamp | 1-10ms | 浣?| USB 鐩告満 |
| ROS ApproximateTime | 1-50ms | 浣?| ROS 绯荤粺 |
| NTP | 1-100ms | 浣?| 閫氱敤缃戠粶 |

---

## 搂5 鈥?澶栧弬鏍囧畾绮惧害璇勪及

### 5.1 璇勪及鎸囨爣

#### 5.1.1 閲嶆姇褰辫宸?(Reprojection Error)

$$e_{\text{reproj}} = \frac{1}{N} \sum_{i=1}^{N} \| \pi(K, D, T_{C}^{W} P_i^W) - \hat{p}_i \|$$

**鍚堟牸鏍囧噯**锛?
- 鍗曠浉鏈烘爣瀹氾細$e_{\text{reproj}} < 0.5$ px
- 鍙岀洰鏍囧畾锛?e_{\text{reproj}} < 0.3$ px锛堟瘡涓浉鏈猴級
- 鐩告満-IMU 鑱斿悎鏍囧畾锛?e_{\text{reproj}} < 0.5$ px

#### 5.1.2 鍩虹嚎闀垮害璇樊

$$e_{\text{baseline}} = | \|t_{c_1}^{c_2}\| - b_{\text{design}} |$$

**鍚堟牸鏍囧噯**锛?e_{\text{baseline}} < 1\%$锛堢浉瀵逛簬璁捐鍊硷級

#### 5.1.3 鏃嬭浆鐭╅樀姝ｄ氦鎬?

$$e_{\text{ortho}} = \|R^T R - I\|_F$$

**鍚堟牸鏍囧噯**锛?e_{\text{ortho}} < 10^{-6}$

### 5.2 绮惧害楠岃瘉瀹為獙

#### 5.2.1 閲嶆姇褰辫宸垎甯冨垎鏋?

```python
import numpy as np

def analyze_reprojection_errors(calibration_result):
    """
    鍒嗘瀽鏍囧畾缁撴灉鐨勯噸鎶曞奖璇樊鍒嗗竷
    """
    errors = calibration_result.reprojection_errors
    
    metrics = {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'max': np.max(errors),
        'median': np.median(errors),
        'rmse': np.sqrt(np.mean(errors**2))
    }
    
    # 妫€鏌ヨ宸垎甯冩槸鍚︽帴杩戞鎬侊紙楠岃瘉鏍囧畾璐ㄩ噺锛?
    from scipy import stats
    _, p_value = stats.normaltest(errors)
    metrics['normality_p'] = p_value
    
    return metrics
```

### 5.3 鏍囧畾绮惧害瀵逛笅娓镐换鍔＄殑褰卞搷

| 鏍囧畾璇樊绫诲瀷 | 瀵?VIO 鐨勫奖鍝?| 瀵规繁搴︿及璁＄殑褰卞搷 | 瀵?SLAM 鐨勫奖鍝?|
|-----------|-------------|--------------|-------------|
| 鍐呭弬璇樊 0.5% | 杞ㄨ抗婕傜Щ +2% | 娣卞害璇樊 +1% | 鍥炵幆妫€娴嬪け璐ョ巼 +5% |
| 澶栧弬璇樊 1掳 | 灏哄害婕傜Щ +5% | 娣卞害鍥捐竟缂橀敊浣?| 鍦板浘鎵洸 |
| 鏃堕棿鍋忕Щ 5ms | 楂橀€熸椂杞ㄨ抗鍙戞暎 | 鍔ㄦ€佺墿浣撴繁搴︿笉鍑?| 浣嶅Э浼拌鎶栧姩 |
| 鍩虹嚎璇樊 1% | 灏哄害璇樊 +1% | 缁濆娣卞害璇樊 +1% | 灏哄害涓嶄竴鑷?|

---

## 搂6 鈥?甯歌鏍囧畾闂涓庤В鍐虫柟妗?

### 6.1 闂璇婃柇鐭╅樀

| 闂鐜拌薄 | 鍙兘鍘熷洜 | 瑙ｅ喅鏂规 |
|---------|---------|---------|
| **閲嶆姇褰辫宸?> 1px** | 鏍囧畾鏉垮钩闈㈠害涓嶅銆侀暅澶寸暩鍙樻ā鍨嬩笉鍖归厤 | 浣跨敤鐜荤拑/閾濆埗鏍囧畾鏉匡紱灏濊瘯鏇撮珮闃剁暩鍙樻ā鍨?|
| **鍙岀洰鏋佺嚎涓嶅榻?* | 澶栧弬鏍囧畾涓嶅噯纭€佸熀绾夸及璁″亸宸?| 澧炲姞鏍囧畾鏁版嵁澶氭牱鎬э紱妫€鏌ユ満姊扮粨鏋勭ǔ瀹氭€?|
| **IMU 鏍囧畾鍙戞暎** | 杩愬姩婵€鍔变笉瓒炽€佺函鏃嬭浆/绾钩绉?| 纭繚 6DoF 杩愬姩锛涘鍔犲姞閫熷害鍜岃閫熷害鍙樺寲 |
| **鏃堕棿鍋忕Щ浼拌涓嶇ǔ瀹?* | 鏁版嵁棰戠巼涓嶅尮閰嶃€佹椂闂存埑鎶栧姩 | 浣跨敤纭欢鍚屾锛涙鏌?ROS 鏃堕棿鎴抽厤缃?|
| **灏哄害浼拌鍋忓樊** | 鏍囧畾鏉垮昂瀵镐笉鍑嗙‘ | 绮剧‘娴嬮噺鏍囧畾鏉垮昂瀵革紱浣跨敤婵€鍏夋祴璺濋獙璇?|
| **Rolling Shutter 鏁堝簲** | 楂橀€熻繍鍔ㄥ鑷村浘鍍忕暩鍙?| 浣跨敤 kalibr_calibrate_rs锛涢檷浣庤繍鍔ㄩ€熷害 |

### 6.2 鏍囧畾鏉垮埗浣滆鑼?

#### AprilGrid 鏍囧畾鏉?

```yaml
# target.yaml
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.0295  # 29.5mm
tagSpacing: 0.01475  # 50% spacing
```

**鍒朵綔瑕佹眰**锛?
- 鎵撳嵃绮惧害锛欴PI >= 600锛堟縺鍏夋墦鍗帮級
- 鏉愯川锛氬搼鍏夌浉绾告垨閾濆鏉匡紙閬垮厤鍙嶅厜锛?
- 骞虫暣搴︼細骞抽潰搴?< 0.1mm
- 灏哄绮惧害锛氬疄闄呭昂瀵镐笌璁捐鍊艰宸?< 0.1%

### 6.3 鏍囧畾鏁版嵁璐ㄩ噺妫€鏌ユ竻鍗?

- [ ] 鐩告満瑙嗛噹鍐呮爣瀹氭澘瑕嗙洊 > 50% 鐨勫浘鍍忓尯鍩?
- [ ] 姣忎釜鐩告満妫€娴嬪埌鏍囧畾鏉跨殑甯ф暟 > 100
- [ ] IMU 鏁版嵁鍖呭惈瓒冲鐨勫姞閫熷害鍙樺寲锛?> 2 m/s^2$锛?
- [ ] IMU 鏁版嵁鍖呭惈瓒冲鐨勮閫熷害鍙樺寲锛?> 100 掳/s$锛?
- [ ] 鏁版嵁鍖呭惈涓嶅悓璺濈锛?.3m - 2m锛夌殑鏍囧畾鏉胯娴?
- [ ] 鏁版嵁鍖呭惈涓嶅悓瑙掑害锛?掳 - 60掳锛夌殑鏍囧畾鏉胯娴?
- [ ] 鏃犺繍鍔ㄦā绯婏紙蹇棬鏃堕棿 < 5ms 鎴栬繍鍔ㄩ€熷害 < 30掳/s锛?

---

## 搂7 鈥?涓?DVAS 椤圭洰鐨勫叧鑱?

### 7.1 鍦ㄦ暟鎹噰闆?Pipeline 涓殑浣嶇疆

```
DVAS 鏁版嵁閲囬泦 Pipeline:

[纭欢缁勮] 鈫?[鑱斿悎鏍囧畾] 鈫?[ ego-centric 閲囬泦] 鈫?[鏁版嵁璐ㄦ] 鈫?[鏍囨敞]
                 鈫?
            Kalibr / Basalt
                 鈫?
        鍙岀洰+IMU 鏍囧畾鍙傛暟
                 鈫?
        [娣卞害浼拌] 鈫?FoundationStereo
        [SLAM] 鈫?ORB-SLAM3 / Kimera
        [鎵嬪娍浼拌] 鈫?HaMeR
```

### 7.2 DVAS 鐗瑰畾鑰冮噺

| DVAS 闇€姹?| 鏍囧畾褰卞搷 | 鎺ㄨ崘鍋氭硶 |
|----------|---------|---------|
| Ego-centric 瑙嗚 | 鐩告満浣╂埓浣嶇疆鍙樺寲闇€閲嶆柊鏍囧畾 | 姣忔浣╂埓鍚庡揩閫熼獙璇佹爣瀹?|
| 闀挎椂闂撮噰闆?(>1h) | IMU bias 婕傜Щ | 瀹氭湡閲嶆柊鏍囧畾鎴栧湪绾挎爣瀹?|
| 澶氳澶囬噰闆?| 璁惧闂存爣瀹氫竴鑷存€?| 缁熶竴鏍囧畾鏉裤€佺粺涓€娴佺▼ |
| 鎵?鐗╀氦浜?| 鎵嬮儴浣嶅Э绮惧害渚濊禆鐩告満鏍囧畾 | 纭繚鎵嬮儴鍖哄煙鐨勯噸鎶曞奖璇樊 < 0.3px |

---

## 搂8 鈥?鍙傝€冧笌璧勬簮

### 8.1 寮€婧愬伐鍏?

| 宸ュ叿 | 閾炬帴 | 閫傜敤鍦烘櫙 |
|------|------|---------|
| Kalibr | https://github.com/ethz-asl/kalibr | 绂荤嚎楂樼簿搴︽爣瀹?|
| Basalt | https://github.com/VladyslavUsenko/basalt-mirror | 鍦ㄧ嚎鏍囧畾銆乂IO |
| OpenCV calib3d | https://docs.opencv.org/ | 鍩虹鐩告満鏍囧畾 |
| CameraIMUCalibration | https://github.com/... | 绠€鍖栫増鐩告満-IMU鏍囧畾 |

### 8.2 鍏抽敭璁烘枃

1. **Rehder et al. (2016)** - "Extending Kalibr: Calibrating the Extrinsics of Multiple IMUs and of Other Associated Sensors"
2. **Furgale et al. (2013)** - "Unified Temporal and Spatial Calibration for Multi-Sensor Systems"
3. **Usenko et al. (2020)** - "The Double Sphere Camera Model"
4. **Nikolic et al. (2014)** - "A Unified Software Framework for Calibration in Robotics"

### 8.3 鐩稿叧鏂囨。

- [娣卞害浼拌锛欶oundationStereo](22-depth-estimation.md) 鈥?渚濊禆绮剧‘鐨勫弻鐩爣瀹?
- [SLAM 绯荤粺](23-slam.md) 鈥?渚濊禆鐩告満-IMU 澶栧弬
- [澶氫紶鎰熷櫒铻嶅悎](26-sensor-fusion.md) 鈥?鏍囧畾鏄浼犳劅鍣ㄨ瀺鍚堢殑鍓嶆彁

---

*Layer: 03-perception | Prev: [鎰熺煡灞傜储寮昡(../INDEX.md) | Next: [娣卞害浼拌](22-depth-estimation.md)*
