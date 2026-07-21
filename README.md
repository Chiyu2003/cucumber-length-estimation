# 🥒 Cucumber 3D Length Estimation (RealSense M5 Only)

本專案提供一套獨立、輕量化的 **Intel RealSense 3D 黃瓜長度與幾何結構自治估算推論引擎**。結合 **YOLO11-seg** 實體分割與 **Method M5 (Medial Arc Spline)** 三維空間骨架弧長擬合演算法，能在無須複雜機器學習迴歸器的情況下，直接從 RGB-D 深度影像中精確計算黃瓜的 3D 空間弧長 (cm)。

---

## 📁 目錄結構說明

```text
m5_realsense_only/
├── data/               # 測試與待測數據 (包含 RGB 影像及 raw depth .npy 深度檔)
├── weights/            # YOLO11-seg 預訓練權重檔 (cucumber_seg.pt)
├── output/             # 渲染結果輸出目錄 (包含遮罩、3D 骨架與長度標籤)
├── utils/              # 核心幾何計算庫 (geometry.py, image_processing.py)
├── inference/          # 獨立單張/批次推論腳本 (predict_m5.py)
├── main.py             # 主推論入口腳本
├── requirements.txt    # Python 套件依賴清單
└── README.md           # 本說明文件
```

---

## ⚙️ 1. 環境設定與安裝 (Environment Setup)

建議使用 **Python 3.10** 以上之虛擬環境：

### 步驟 A：建立並啟用虛擬環境
```bash
# 使用 venv 建立虛擬環境
python3 -m venv rs_env
source rs_env/bin/activate  # macOS / Linux
# rs_env\Scripts\activate   # Windows
```

### 步驟 B：安裝套件依賴
```bash
pip install -r requirements.txt
```

> **主要依賴項清單**：
> * `ultralytics` (YOLO11-seg 推論)
> * `opencv-python` (圖像處裡與渲染)
> * `numpy` (陣列與點雲運算)
> * `scipy` (3D Cubic Spline 曲線插值與微積分)
> * `matplotlib` (視覺化)

---

## 🧪 2. 測試資料說明 (Available Test Data)

`data/` 目錄中預先為您準備了 **20 張完全未參與過 YOLO 訓練的獨立溫室測試快照 (Unseen Test Data)**，供您即時驗證推論：

* **中段時段快照 (10 張)**：`snapshot_20260617_115658...` 至 `125407`（包含標準姿態與不同光照）
* **後段時段快照 (10 張)**：`snapshot_20260617_125920...` 至 `130904`（包含斜放姿態與雙黃瓜場景）
* **對應深度檔**：每張 `.png` 均附帶對應檔名的 `_depth.npy` Raw Depth 深度陣列 (單位 mm)。

### 💡 如何新增自己的測試資料？
只需將您的彩色照片與對應的 RealSense 深度圖檔放進 `data/` 目錄：
1. 彩色照片：`your_image_name.png` (或 `.jpg`)
2. 深度圖檔：`your_image_name_depth.npy` (必須為 NumPy 2D 陣列)

---

## 🚀 3. 執行推論 (Execution)

### 批次執行 `data/` 目錄下所有測試資料：
```bash
python main.py
```

### 輸出成果：
執行完成後，自動於 `output/` 目錄下生成帶有視覺化 Overlay 的影像：
* 🟢 **綠色遮罩**：YOLO11-seg 實體切割輪廓
* 🔴 **紅色包圍盒**：NMS 過濾後的黃瓜 Bounding Box
* 🟡 **黃色 3D 骨架線**：經 3D 平面趨勢過濾與 Cubic Spline 擬合的最精確中心線
* 🏷️ **黃色文字**：最終估算之 3D 弧長（單位 `cm`）

---

## 🛠️ 4. 核心演算法亮點 (Key Algorithm Features)

1. **3D Plane Trend Depth Fitting**：採用三維空間趨勢平面過濾，解決黃瓜斜放、傾斜時頭端被誤剔除的痛點。
2. **IoU Non-Maximum Suppression (NMS)**：內建 `filter_duplicate_boxes` 邊框過濾，徹底排除同根黃瓜重疊誤判。
3. **Refined Tip Quantile (0.2%)**：極限鎖定黃瓜頭尾端點，確保 3D Spline 100% 延伸涵蓋至瓜體最外側邊界。
