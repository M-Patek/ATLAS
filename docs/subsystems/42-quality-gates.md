---
id: quality-gates
title: "Quality Gates: Data Quality Assurance"
status: complete
complexity: medium
related:
  - "../05-integration/01-pipeline-patterns.md"
  - "02-action-annotation.md"
  - "../03-perception/01-stereo-imu.md"
  - "../04-data-ecosystem/06-data-formats.md"
prerequisites:
  - "Data Pipeline Patterns"
  - "Annotation Standards"
  - "Basic Statistics"
last_validated: 2026-06-27
---

# Quality Gates: 鏁版嵁璐ㄩ噺闂ㄦ帶绛栫暐

## 搂0 鈥?One-liner

Embodied AI 鏁版嵁璐ㄩ噺鐨勫叏鐢熷懡鍛ㄦ湡淇濋殰鈥斺€斾粠閲囬泦鍒拌缁冪殑澶氱淮搴﹁川閲忔娴嬨€佽嚜鍔ㄩ棬鎺т笌鍙嶉闂幆銆?

## 搂1 鈥?Concept Map

```mermaid
graph TB
    subgraph "Quality Dimensions"
        Q1[瀹屾暣鎬
        Q2[涓€鑷存€
        Q3[鍑嗙‘鎬
        Q4[鏃舵晥鎬
    end

    subgraph "Detection Methods"
        D1[鑷姩妫€娴媇
        D2[浜哄伐鎶芥]
        D3[浜ゅ弶楠岃瘉]
    end

    subgraph "Feedback Loop"
        F1[闂鍙戠幇]
        F2[鏍瑰洜瀹氫綅]
        F3[淇楠岃瘉]
        F4[鎸囨爣鏇存柊]
    end

    Q1 --> D1
    Q2 --> D1
    Q3 --> D2
    Q4 --> D1
    D1 --> F1
    D2 --> F1
    F3 --> D3
    F4 --> Q1
```

## 搂2 鈥?Data Quality Dimensions

### 2.1 鍥涚淮搴︽鏋?

embodied AI 鏁版嵁璐ㄩ噺闇€瑕佷粠鍥涗釜鏍稿績缁村害杩涜璇勪及锛屾瘡涓淮搴﹀搴斾笉鍚岀殑妫€娴嬬瓥鐣ュ拰宸ュ叿銆?

| 缁村害 | 瀹氫箟 | 鍏抽敭鎸囨爣 | 妫€娴嬮鐜?|
|------|------|----------|----------|
| **瀹屾暣鎬?* | 鏁版嵁璁板綍鏄惁榻愬叏锛屾棤缂哄け | 缂哄け鐜囥€佽鐩栫巼銆佸瓧娈靛～鍏呯巼 | 姣忔壒娆?|
| **涓€鑷存€?* | 璺ㄦā鎬併€佽法鏃堕棿鐨勬暟鎹榻?| 鏃堕棿鎴冲亸宸€佷紶鎰熷櫒鍚屾璇樊 | 瀹炴椂 |
| **鍑嗙‘鎬?* | 鏍囨敞涓庣湡瀹炵姸鎬佺殑涓€鑷寸▼搴?| 鏍囨敞鍑嗙‘鐜囥€両oU銆丮PJPE | 姣忔壒娆?鎶芥 |
| **鏃舵晥鎬?* | 鏁版嵁鏂伴矞搴︿笌鐗堟湰鏈夋晥鎬?| 閲囬泦-澶勭悊寤惰繜銆佹ā鍨嬬増鏈尮閰?| 瀹炴椂 |

### 2.2 瀹屾暣鎬?(Completeness)

embodied AI 鏁版嵁鐨勫畬鏁存€ф秹鍙婂妯℃€佸悓姝ラ噰闆嗙殑瀹屾暣鎬т繚闅溿€?

**浼犳劅鍣ㄦ暟鎹畬鏁存€ф鏌ユ竻鍗?*

| 妫€鏌ラ」 | 鏂规硶 | 闃堝€肩ず渚?|
|--------|------|----------|
| 鐩告満甯у簭鍒楄繛缁?| 甯у彿宸垎妫€娴?| 涓㈠抚鐜?< 0.1% |
| IMU 鏁版嵁瑕嗙洊 | 鏃堕棿鎴冲尯闂存瘮瀵?| 瑕嗙洊鐜?> 99.5% |
| 娣卞害鍥惧榻?| RGB-D 鏃堕棿鎴冲樊 | < 5ms |
| 鍔ㄤ綔杞ㄨ抗瀹屾暣 | 鍏抽敭甯ф娴?| 璧锋鐘舵€佸畬鏁?|
| 鍏冩暟鎹瓧娈靛～鍏?| Schema 鏍￠獙 | 蹇呭～瀛楁 100% |

**甯歌缂哄け妯″紡涓庢牴鍥?*

```
缂哄け妯″紡                          鏍瑰洜鍒嗘瀽
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
鐩告満甯у懆鏈熸€т涪澶?(姣?0甯т涪1甯?    鈫?USB 甯﹀鐡堕 / 缂撳啿鍖烘孩鍑?
IMU 鏁版嵁娈电己澶?                   鈫?浼犳劅鍣ㄤ复鏃舵柇寮€ / 鐢垫簮鎶栧姩
娣卞害鍥句笌 RGB 涓嶅悓姝?              鈫?纭欢瑙﹀彂淇″彿婕傜Щ
鍔ㄤ綔鏍囩缂哄け                      鈫?鏍囨敞宸ュ叿宕╂簝 / 浜哄伐璺宠繃
鍏冩暟鎹瓧娈典负绌?                   鈫?閲囬泦杞欢鐗堟湰涓嶅尮閰?
```

### 2.3 涓€鑷存€?(Consistency)

澶氭ā鎬佹暟鎹殑涓€鑷存€?embodied AI 涓渶鍏抽敭鐨勮川閲忕淮搴︺€?

**鏃堕棿涓€鑷存€?*

```python
# 鏃堕棿鍚屾鏍￠獙浼唬鐮?
def verify_temporal_consistency(frames, imu_data, action_labels):
    issues = []

    # 1. 鐩告満鍐呴儴甯х巼涓€鑷存€?
    frame_intervals = np.diff(frames.timestamps)
    if np.std(frame_intervals) > 0.1 * np.mean(frame_intervals):
        issues.append("Variable frame rate detected")

    # 2. 璺ㄦā鎬佹椂闂存埑瀵归綈
    rgb_ts = frames.rgb_timestamps
    depth_ts = frames.depth_timestamps
    imu_ts = imu_data.timestamps

    cross_modal_drift = max(
        np.max(np.abs(rgb_ts - depth_ts)),
        np.max(np.abs(rgb_ts[:len(imu_ts)] - imu_ts))
    )
    if cross_modal_drift > 5:  # ms
        issues.append(f"Cross-modal drift: {cross_modal_drift}ms")

    # 3. 鍔ㄤ綔鏍囩涓庤棰戞椂闂村榻?
    for label in action_labels:
        if label.start_time < frames.start_time or label.end_time > frames.end_time:
            issues.append(f"Action label out of video bounds: {label.id}")

    return issues
```

**绌洪棿涓€鑷存€?*

| 妫€鏌ラ」 | 鎻忚堪 | 鍏稿瀷闃堝€?|
|--------|------|----------|
| 鐩告満-IMU 澶栧弬涓€鑷存€?| Kalibr 鏍囧畾缁撴灉涓庡嚭鍘傚弬鏁板亸宸?| < 2% |
| 娣卞害- RGB 閲嶆姇褰辫宸?| 娣卞害鍥炬姇褰卞埌 RGB 骞抽潰 | < 2 pixels |
| 鎵嬪Э-鐗╀綋鎺ヨЕ涓€鑷存€?| 鎵嬮儴涓庣墿浣撹窛绂?vs 鏍囨敞鎺ヨЕ鐘舵€?| 閫昏緫涓€鑷存€?|
| SLAM 杞ㄨ抗杩炵画鎬?| 浣嶅Э璺宠穬妫€娴?| < 5cm / frame |

### 2.4 鍑嗙‘鎬?(Accuracy)

鍑嗙‘鎬ц瘎浼伴渶瑕?ground truth 鎴栭珮缃俊搴﹀弬鑰冦€?

**鑷姩鏍囨敞鍑嗙‘鎬ц瘎浼?*

| 妯℃€?| 璇勪及鏂规硶 | 鍙傝€冩爣鍑?|
|------|----------|----------|
| 鐗╀綋妫€娴?| mAP @ IoU=0.5 | 浜哄伐鏍囨敞瀛愰泦 |
| 鎵嬪Э浼拌 | MPJPE (Mean Per Joint Position Error) | MoCap 绯荤粺 |
| 娣卞害浼拌 | RMSE, AbsRel | LiDAR 鐐逛簯 |
| 鍔ㄤ綔鍒嗗壊 | F1-score, Frame-wise Accuracy | 浜哄伐鍒嗘 |
| 鍦烘櫙鐞嗚В | 璇箟鍒嗗壊 mIoU | 浜哄伐鏍囨敞 |

**鍑嗙‘鎬у垎绾ф爣鍑?*

| 绛夌骇 | 娣卞害浼拌 RMSE | 鎵嬪Э MPJPE | 鐗╀綋妫€娴?mAP | 鐢ㄩ€?|
|------|-------------|------------|--------------|------|
| A (Research) | < 0.05m | < 5mm | > 0.95 | 璁烘枃瀹為獙 |
| B (Production) | < 0.1m | < 10mm | > 0.90 | 鐢熶骇璁粌 |
| C (Pretraining) | < 0.2m | < 20mm | > 0.80 | 澶ц妯￠璁粌 |
| D (Filtered) | > 0.2m | > 20mm | < 0.80 | 闇€浜哄伐鏍￠獙鎴栦涪寮?|

### 2.5 鏃舵晥鎬?(Timeliness)

embodied AI 鏁版嵁鐨勬椂鏁堟€у奖鍝嶆ā鍨嬪褰撳墠鐜鐨勯€傚簲鑳藉姏銆?

| 鎸囨爣 | 瀹氫箟 | 鐩爣鍊?|
|------|------|--------|
| 閲囬泦-澶勭悊寤惰繜 | 鍘熷鏁版嵁鍒板彲鐢ㄨ缁冩牱鏈殑鏃堕棿 | < 24h (鎵瑰鐞? |
| 妯″瀷鐗堟湰鍖归厤 | 璁粌鏁版嵁涓庨儴缃叉ā鍨嬬殑鐗堟湰鍏煎鎬?| 鑷姩鏍￠獙 |
| 鐜婕傜Щ妫€娴?| 褰撳墠鏁版嵁涓庤缁冨垎甯冪殑鍋忕搴?| 鎸佺画鐩戞帶 |
| 鏁版嵁淇濊川鏈?| 鐗瑰畾鍦烘櫙鏁版嵁鐨勬湁鏁堟湡 | 鐜鍙樺寲瑙﹀彂鏇存柊 |

## 搂3 鈥?Automated Detection

### 3.1 寮傚父鍊兼娴?(Outlier Detection)

**缁熻鏂规硶**

| 鏂规硶 | 閫傜敤鍦烘櫙 | 浼樼偣 | 灞€闄?|
|------|----------|------|------|
| Z-Score / IQR | 鍗曞彉閲忔暟鍊肩壒寰?| 绠€鍗曢珮鏁?| 鍋囪楂樻柉鍒嗗竷 |
| Isolation Forest | 澶氱淮搴﹁仈鍚堝紓甯?| 鏃犻渶鍒嗗竷鍋囪 | 楂樼淮鏁堟灉涓嬮檷 |
| LOF (Local Outlier Factor) | 灞€閮ㄥ瘑搴﹀紓甯?| 鍙戠幇绨囧唴寮傚父 | 璁＄畻澶嶆潅搴﹂珮 |
| 鍩轰簬 VAE 鐨勯噸鏋勮宸?| 鍥惧儚/娣卞害鍥惧紓甯?| 鎹曡幏璇箟寮傚父 | 闇€璁粌鏁版嵁 |

**embodied AI 涓撶敤寮傚父妯″紡**

```
瑙嗚寮傚父:
  - 杩囨洕/娆犳洕: 鐩存柟鍥炬瀬鍖栨娴?
  - 杩愬姩妯＄硦: 鎷夋櫘鎷夋柉绠楀瓙鏂瑰樊 < threshold
  - 闀滃ご閬尅: 澶ч潰绉潎鍖€鍖哄煙妫€娴?
  - 閿欒鐧藉钩琛? 鐏颁笘鐣屽亣璁惧亸绂?

IMU 寮傚父:
  - 楗卞拰: 鏁板€艰Е鍙婇噺绋嬭竟鐣?
  - 闆舵紓: 闈欐鐘舵€佸潎鍊煎亸绉?
  - 楂橀鍣０: FFT 棰戣氨寮傚父

鍔ㄤ綔寮傚父:
  - 鍏宠妭闄愪綅: 瓒呭嚭鏈烘/鐢熺悊鑼冨洿
  - 閫熷害绐佸彉: 鐩搁偦甯ч€熷害鍙樺寲 > 鐗╃悊鍙兘
  - 鑷鎾? 杩愬姩瀛︽鏌ュけ璐?
```

### 3.2 鍒嗗竷婕傜Щ妫€娴?(Distribution Drift)

褰撻儴缃茬幆澧冩垨閲囬泦鏉′欢鍙樺寲鏃讹紝鏁版嵁鍒嗗竷鍙兘鍙戠敓婕傜Щ锛屽奖鍝嶆ā鍨嬫€ц兘銆?

**婕傜Щ绫诲瀷**

| 绫诲瀷 | 鎻忚堪 | 妫€娴嬫柟娉?|
|------|------|----------|
| 鍗忓彉閲忔紓绉?(Covariate Shift) | P(X) 鍙樺寲锛孭(Y\|X) 涓嶅彉 | 鐗瑰緛鍒嗗竷 KS-test |
| 姒傚康婕傜Щ (Concept Drift) | P(Y\|X) 鍙樺寲 | 妯″瀷缃俊搴︿笅闄嶆娴?|
| 鏍囩婕傜Щ (Label Shift) | P(Y) 鍙樺寲 | 鏍囩鍒嗗竷鐩戞帶 |

**瀹炵敤妫€娴嬫柟妗?*

```python
class DriftDetector:
    def __init__(self, reference_data, window_size=1000):
        self.reference = reference_data
        self.window = deque(maxlen=window_size)
        self.threshold = 0.05  # p-value

    def check_feature_drift(self, new_sample):
        """鍩轰簬 KS-test 鐨勫崟鐗瑰緛婕傜Щ妫€娴?""
        self.window.append(new_sample)
        if len(self.window) < self.window.maxlen:
            return False, None

        drift_detected = False
        reports = []
        for feature in self.reference.columns:
            stat, p_value = ks_2samp(
                self.reference[feature],
                [s[feature] for s in self.window]
            )
            if p_value < self.threshold:
                drift_detected = True
                reports.append({
                    'feature': feature,
                    'ks_statistic': stat,
                    'p_value': p_value
                })

        return drift_detected, reports

    def check_embedding_drift(self, new_embeddings):
        """鍩轰簬宓屽叆绌洪棿璺濈鐨勮涔夋紓绉绘娴?""
        ref_mean = np.mean(self.reference_embeddings, axis=0)
        ref_cov = np.cov(self.reference_embeddings.T)

        # 椹皬璺濈
        inv_cov = np.linalg.inv(ref_cov)
        distances = []
        for emb in new_embeddings:
            diff = emb - ref_mean
            d = np.sqrt(diff.T @ inv_cov @ diff)
            distances.append(d)

        # 瓒呰繃 3-sigma 瑙嗕负婕傜Щ
        return np.mean(distances) > 3.0
```

### 3.3 鏃堕棿鎴虫牎楠?

鏃堕棿鎴虫槸 embodied AI 鏁版嵁涓€鑷存€х殑鍩虹煶銆?

**鏍￠獙灞傜骇**

| 灞傜骇 | 鏍￠獙鍐呭 | 宸ュ叿 |
|------|----------|------|
| L1: 鍗曡皟鎬?| 鏃堕棿鎴充弗鏍奸€掑 | 绠€鍗曞樊鍒?|
| L2: 鑼冨洿 | 钀藉湪鍚堢悊鏃堕棿鍖洪棿 | 杈圭晫妫€鏌?|
| L3: 绮惧害 | 鏃堕棿鎴崇簿搴︾鍚堥鏈?(ms/us/ns) | 灏忔暟浣嶅垎鏋?|
| L4: 鍚屾 | 澶氫紶鎰熷櫒鏃堕棿鎴冲榻?| 浜掔浉鍏冲垎鏋?|
| L5: 鏃堕挓婧?| NTP/PTP 鍚屾鐘舵€?| 绯荤粺鏃ュ織 |

**甯歌鏃堕棿鎴抽棶棰?*

```
闂鐜拌薄                          鏍瑰洜                          淇
鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
鏃堕棿鎴冲€掑簭 (t[i+1] < t[i])        缂撳啿鍖洪噸鎺掑簭 / NTP 璺冲彉          鍗曡皟鎬у己鍒?+ 鏃ュ織鍛婅
寰绾х簿搴︿絾姣绾у彉鍖?           绮惧害澹版槑閿欒                     缁熶竴绮惧害鏍煎紡
澶氱浉鏈烘椂闂存埑绯荤粺鎬у亸绉?           瑙﹀彂淇″彿浼犳挱寤惰繜                  纭欢鍚屾 (PTP/Genlock)
ROS bag 鏃堕棿鎴充笌 wall-clock 鍋忓樊   褰曞埗鏃剁郴缁熻礋杞介珮                  浣跨敤 /clock topic
璺ㄤ細璇濇椂闂存埑涓嶈繛缁?               閲囬泦涓柇                         session 杈圭晫鏍囪
```

## 搂4 鈥?Manual Sampling Inspection

### 4.1 鎶芥牱绛栫暐

鑷姩妫€娴嬫棤娉曡鐩栨墍鏈夎川閲忛棶棰橈紝浜哄伐鎶芥鏄噯纭€х殑鏈€缁堜繚闅溿€?

| 绛栫暐 | 鏂规硶 | 閫傜敤鍦烘櫙 |
|------|------|----------|
| 闅忔満鎶芥牱 | 鍧囧寑闅忔満閫夊彇 | 鍩虹嚎璐ㄩ噺璇勪及 |
| 鍒嗗眰鎶芥牱 | 鎸変换鍔?鍦烘櫙鍒嗗眰鍚庨殢鏈?| 纭繚鍚勭被鍒鐩?|
| 缃俊搴﹀姞鏉?| 浣庣疆淇″害鏍锋湰鏇撮珮琚娊涓鐜?| 鑱氱劍闂鍖哄煙 |
| 鑱氱被鎶芥牱 | 宓屽叆绌洪棿鑱氱被鍚庢瘡绫绘娊鏍?| 鍙戠幇杈圭紭妗堜緥 |
| 鏃跺簭鎶芥牱 | 姣?N 绉?甯ф娊鏍?| 瑙嗛鏍囨敞鏍￠獙 |

**鎶芥牱姣斾緥寤鸿**

| 鏁版嵁闃舵 | 鑷姩妫€娴嬭鐩栫巼 | 浜哄伐鎶芥姣斾緥 | 璇存槑 |
|----------|---------------|-------------|------|
| 鍘熷閲囬泦 | 90% | 1-2% | 涓昏渚濊禆鑷姩鏍￠獙 |
| 鑷姩鏍囨敞杈撳嚭 | 85% | 5-10% | 鑷姩鏍囨敞閿欒鐜囪緝楂?|
| 浜哄伐鏍囨敞杈撳嚭 | 95% | 10-20% | 璐ㄦ鍛樻娊妫€ |
| 鏈€缁堣缁冮泦 | 98% | 2-5% | 缁煎悎璐ㄩ噺纭 |

### 4.2 鏍囨敞涓€鑷存€ф楠?

澶氫汉鏍囨敞鍦烘櫙涓嬬殑涓€鑷存€ф槸鍏抽敭璐ㄩ噺鎸囨爣銆?

**涓€鑷存€у害閲?*

| 鏍囨敞绫诲瀷 | 涓€鑷存€ф寚鏍?| 鍙帴鍙楅槇鍊?|
|----------|-----------|-----------|
| 杈圭晫妗?| IoU 閲嶅彔鐜?| > 0.85 |
| 鍏抽敭鐐?| OKS (Object Keypoint Similarity) | > 0.75 |
| 璇箟鍒嗗壊 | mIoU | > 0.80 |
| 鍔ㄤ綔鍒嗘 | 璧锋鏃堕棿宸?| < 200ms |
| 鍔ㄤ綔绫诲埆 | Cohen's Kappa | > 0.80 |

**鎻愬崌涓€鑷存€х殑鏂规硶**

1. **鏍囨敞鎸囧崡缁嗗寲**: 鎻愪緵杈圭晫妗堜緥 (edge cases) 鍜屽喅绛栨爲
2. **鏍″噯浼氳瘽 (Calibration Session)**: 姝ｅ紡鏍囨敞鍓嶇粺涓€鐞嗚В
3. **杩唬鍙嶉**: 瀹氭湡鍥為【鍒嗘妗堜緥锛屾洿鏂版寚鍗?
4. **榛勯噾鏍囧噯闆?*: 缁存姢宸茬煡绛旀鐨勫弬鑰冮泦锛屽畾鏈熸祴璇曟爣娉ㄥ憳

## 搂5 鈥?Feedback Loop: 璐ㄩ噺闂杩芥函涓庝慨澶?

### 5.1 闂幆鏋舵瀯

```mermaid
graph LR
    A[璐ㄩ噺妫€娴嬪彂鐜伴棶棰榏 --> B{闂鍒嗙骇}
    B -->|P0: 闃绘柇鎬 C[绔嬪嵆鍋滄娴佹按绾縘
    B -->|P1: 涓ラ噸| D[鏍囪闅旂 + 鍛婅]
    B -->|P2: 涓€鑸瑋 E[璁板綍寰呬慨澶嶉槦鍒梋
    B -->|P3: 杞诲井| F[鏃ュ織璁板綍]

    C --> G[鏍瑰洜鍒嗘瀽]
    D --> G
    E --> H[鎵归噺淇]
    F --> I[瓒嬪娍鐩戞帶]

    G --> J[淇楠岃瘉]
    H --> J
    J --> K[鍥炲綊娴嬭瘯]
    K --> L[鏇存柊璐ㄩ噺鎸囨爣]
    L --> M[鐭ヨ瘑搴撴洿鏂癩
```

### 5.2 闂鍒嗙骇鏍囧噯

| 绾у埆 | 鎻忚堪 | 鍝嶅簲鏃堕棿 | 绀轰緥 |
|------|------|----------|------|
| P0 - 闃绘柇 | 瀵艰嚧鏁版嵁涓嶅彲鐢ㄦ垨妯″瀷璁粌澶辫触 | 绔嬪嵆 (< 1h) | 澶ц妯℃椂闂存埑閿欎贡銆佷紶鎰熷櫒澶辨晥 |
| P1 - 涓ラ噸 | 鏄捐憲褰卞搷鏁版嵁璐ㄩ噺锛岄渶闅旂澶勭悊 | 4h | 娣卞害浼拌绯荤粺鎬у亸宸€佹爣娉ㄥぇ闈㈢Н閿欒 |
| P2 - 涓€鑸?| 灞€閮ㄨ川閲忛棶棰橈紝鍙壒閲忎慨澶?| 24h | 涓埆 session 鏍囨敞涓嶄竴鑷淬€佸厓鏁版嵁缂哄け |
| P3 - 杞诲井 | 涓嶅奖鍝嶆牳蹇冨姛鑳斤紝璁板綍鐩戞帶 | 涓嬫杩唬 | 鏍煎紡涓嶈鑼冦€侀潪鍏抽敭瀛楁缂哄け |

### 5.3 鏍瑰洜鍒嗘瀽妯℃澘

```yaml
# 璐ㄩ噺闂 RCA 妯℃澘
issue_id: QG-2024-001
severity: P1
detected_at: "2024-03-15T08:23:00Z"
detection_method: "distribution_drift_alert"

symptom:
  description: "娣卞害浼拌缁撴灉鍦?session-045 涓郴缁熸€у亸澶?15%"
  affected_data: "session-045 to session-052"
  affected_samples: 12450

root_cause:
  category: "calibration_drift"
  description: "Kalibr 鏍囧畾鍚庣浉鏈哄唴鍙傛湭姝ｇ‘鍐欏叆 metadata"
  responsible_component: "calibration_pipeline@v2.3"

fix:
  action: "閲嶆柊鏍囧畾骞舵洿鏂?metadata"
  commit: "abc123"
  validated_by: "depth_rmse < 0.05m"

prevention:
  - "鏍囧畾缁撴灉鑷姩鏍￠獙: 鍐呭弬鍚堢悊鎬ф鏌?
  - "metadata 瀹屾暣鎬?gate: 蹇呭～瀛楁鏍￠獙"
```

## 搂6 鈥?Quality Metrics Definition & Monitoring

### 6.1 鏍稿績璐ㄩ噺鎸囨爣浠〃鏉?

| 灞傜骇 | 鎸囨爣 | 璁＄畻鏂瑰紡 | 鍛婅闃堝€?|
|------|------|----------|----------|
| 閲囬泦灞?| 浼犳劅鍣ㄥ彲鐢ㄧ巼 | 鏈夋晥甯ф暟 / 鏈熸湜甯ф暟 | < 99% |
| 閲囬泦灞?| 鏃堕棿鍚屾璇樊 | 璺ㄦā鎬佹椂闂存埑鏈€澶у樊 | > 5ms |
| 鎰熺煡灞?| 娣卞害浼拌绮惧害 | RMSE vs LiDAR | > 0.1m |
| 鎰熺煡灞?| SLAM 杞ㄨ抗璐ㄩ噺 | 鍥炵幆闂悎璇樊 | > 0.05m |
| 鏍囨敞灞?| 鑷姩鏍囨敞缃俊搴?| 骞冲潎缃俊搴﹀垎鏁?| < 0.85 |
| 鏍囨敞灞?| 浜哄伐涓€鑷存€?| Cohen's Kappa | < 0.80 |
| 鏁版嵁闆?| 鍒嗗竷婕傜Щ鎸囨暟 | KL divergence vs reference | > 0.1 |
| 鏁版嵁闆?| 绫诲埆骞宠　搴?| 鏈€灏忕被 / 鏈€澶х被姣斾緥 | < 0.1 |

### 6.2 鐩戞帶瀹炵幇

```python
# 璐ㄩ噺鎸囨爣涓婃姤绀轰緥
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class QualityMetric:
    name: str
    value: float
    unit: str
    timestamp: float
    threshold: Optional[float] = None
    severity: str = "info"  # info, warning, critical

class QualityMonitor:
    def __init__(self, exporter):
        self.exporter = exporter  # Prometheus / Grafana / MLflow
        self.metrics_history = {}

    def record(self, metric: QualityMetric):
        self.exporter.emit(metric)
        self.metrics_history.setdefault(metric.name, []).append(metric)

        # 闃堝€兼鏌?
        if metric.threshold and metric.value > metric.threshold:
            self._alert(metric)

    def _alert(self, metric: QualityMetric):
        alert = {
            "metric": metric.name,
            "value": metric.value,
            "threshold": metric.threshold,
            "timestamp": metric.timestamp,
            "message": f"Quality gate breached: {metric.name}={metric.value}"
        }
        self.exporter.alert(alert)

# 浣跨敤绀轰緥
monitor = QualityMonitor(prometheus_exporter)
monitor.record(QualityMetric(
    name="depth_estimation_rmse",
    value=0.12,
    unit="m",
    timestamp=time.time(),
    threshold=0.1,
    severity="warning"
))
```

### 6.3 璐ㄩ噺鎶ュ憡鑷姩鐢熸垚

```yaml
# 姣忔棩璐ㄩ噺鎶ュ憡妯℃澘
report_date: "2024-03-15"
dataset: "ego-kitchen-v2"

summary:
  total_sessions: 156
  total_frames: 2,340,000
  pass_rate: 94.2%
  issues_found: 9

dimension_scores:
  completeness: 98.5  # 瀛楁濉厖鐜?
  consistency: 96.3    # 鏃堕棿鍚屾鍚堟牸鐜?
  accuracy: 91.7       # 鏍囨敞涓€鑷存€?
  timeliness: 99.1     # 澶勭悊寤惰繜杈炬爣鐜?

top_issues:
  - issue: "娣卞害浼拌 RMSE 鍋忛珮 (session-045~052)"
    severity: P1
    status: resolved
    fix: "閲嶆柊鏍囧畾"

  - issue: "鎵嬪Э鏍囨敞涓€鑷存€т笅闄?
    severity: P2
    status: in_progress
    action: "鏇存柊鏍囨敞鎸囧崡 + 鏍″噯浼氳瘽"

drift_indicators:
  covariate_drift: false
  concept_drift: false
  label_shift: "mild (+5% 'grasp' actions)"

recommendations:
  - "澧炲姞 'pour' 绫诲埆鏍锋湰 (褰撳墠鍗犳瘮 3%)"
  - "妫€鏌?wrist-camera-03 瀵圭劍鐘舵€?
```

## 搂7 鈥?DVAS 椤圭洰鍏宠仈

DVAS 妗嗘灦涓紝Quality Gates 鏄暟鎹彲淇″害鐨勬牳蹇冧繚闅滄満鍒讹細

| DVAS 鑳藉姏 | Quality Gate 瀹炵幇 |
|-----------|------------------|
| 鏁版嵁鍙俊搴﹁瘎鍒?| 鍥涚淮搴﹀姞鏉冪患鍚堝緱鍒?|
| 闂鍙拷婧?| RCA 妯℃澘 + 鏁版嵁琛€缂?|
| 鑷姩淇 | 鍒嗙骇鍝嶅簲 + 棰勮淇娴佺▼ |
| 鎸佺画鏀硅繘 | 璐ㄩ噺瓒嬪娍鐩戞帶 + 鏍瑰洜鐭ヨ瘑搴?|
| 鍚堣瀹¤ | 瀹屾暣璐ㄩ噺鎶ュ憡鍘嗗彶 |

## 搂8 鈥?Tooling Recommendations

| 鐢ㄩ€?| 鎺ㄨ崘宸ュ叿 | 澶囬€?|
|------|----------|------|
| 鏁版嵁璐ㄩ噺妗嗘灦 | Great Expectations | Soda Core, Deequ |
| 寮傚父妫€娴?| PyOD, Alibi Detect | Isolation Forest (sklearn) |
| 婕傜Щ鐩戞帶 | Evidently AI | WhyLogs, Fiddler |
| 鎸囨爣鐩戞帶 | Prometheus + Grafana | MLflow, Weights & Biases |
| 鍛婅閫氱煡 | PagerDuty + Slack | OpsGenie, 浼佷笟寰俊 |
| 璐ㄩ噺鎶ュ憡 | Jupyter + Papermill | Apache Superset |

## 搂9 鈥?Best Practices Checklist

- [ ] **閲囬泦闃舵**: 浼犳劅鍣ㄥ仴搴锋鏌ヨ嚜鍔ㄥ寲锛屽紓甯哥珛鍗冲憡璀?
- [ ] **棰勫鐞嗛樁娈?*: 鏃堕棿鍚屾鏍￠獙浣滀负纭?gate锛屽け璐ュ嵆闃绘柇
- [ ] **鎰熺煡闃舵**: 姣忔壒娆¤緭鍑虹簿搴︽姤鍛婏紝浣庝簬闃堝€艰Е鍙戜汉宸ュ鏍?
- [ ] **鏍囨敞闃舵**: 鑷姩鏍囨敞蹇呴』缁忚繃缃俊搴﹁繃婊?+ 浜哄伐鎶芥
- [ ] **闆嗘垚闃舵**: 鏈€缁堟暟鎹泦閫氳繃瀹屾暣璐ㄩ噺鎶ュ憡锛屾柟鍙繘鍏ヨ缁?
- [ ] **鐩戞帶闃舵**: 璐ㄩ噺鎸囨爣瀹炴椂涓婃澘锛岃秼鍔垮紓甯歌嚜鍔ㄥ憡璀?
- [ ] **鍙嶉闃舵**: 姣忎釜 P0/P1 闂蹇呴』瀹屾垚 RCA 骞舵洿鏂伴闃叉帾鏂?

---

*Related: [01-pipeline-patterns](41-pipeline-patterns.md) | [03-system-design](43-system-design.md) | Prev: [02-annotation](../INDEX.md)*
