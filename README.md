# Cucumber 3D Length Estimation

本專案提供一套獨立、輕量化的 **Intel RealSense 3D 黃瓜長度與幾何結構自治估算推論引擎**。結合 **YOLO11-seg** 實體分割與 **Medial Arc Spline** 三維空間骨架弧長擬合演算法，直接從 RGB-D 深度影像中精確計算黃瓜的 3D 空間弧長 (cm)。

---

## 目錄結構說明

```text
m5_realsense_only/
├── data/               # 測試與待測數據
├── weights/            # YOLO11-seg 預訓練權重檔
├── output/             # 渲染結果輸出目錄 (包含遮罩、3D 骨架與長度標籤)
├── utils/              # 核心幾何計算
├── inference/          # 獨立單張/批次推論腳本 (predict_m5.py)
├── main.py             # 主推論入口腳本
├── requirements.txt    # Python 套件依賴清單
└── README.md           # 本說明文件
```

---

## 1. 環境設定與安裝 (Environment Setup)

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

---

## 2. 測試資料說明 (Available Test Data)

`data/` 目錄中預先為你準備了 **30 張未參與過 YOLO 訓練的獨立測試資料，供你即時驗證推論：

* **對應深度檔**：每張 `.png` 均附帶對應檔名的 `_depth.npy` Raw Depth 深度陣列 (單位 mm)。
* **品質保證**：已 100% 過濾掉 No Mask 與幾何失敗樣本，確保每一張均可成功預估長度。

### 如何新增自訂的測試資料？
只需將您的彩色照片與對應的 RealSense 深度圖檔放進 `data/` 目錄：
1. 彩色照片：`your_image_name.png` (或 `.jpg`)
2. 深度圖檔：`your_image_name_depth.npy` (必須為 NumPy 2D 陣列，單位 mm 或 m)
3. 如使用不同相機，請於 `utils/image_processing.py` 或 `main.py` 中調整相機內參數 (`FX`, `FY`, `CX`, `CY`)，確保深度圖與彩色圖精確對齊。

---

## 3. 執行推論 (Execution)

### 批次執行 `data/` 目錄下所有測試資料：
```bash
python main.py
```

### 輸出成果說明：
執行完成後，自動於 `output/` 目錄下生成帶有視覺化 Overlay 的影像：
* **綠色遮罩**：YOLO11-seg 實體切割輪廓
* **紅色包圍盒**：NMS 過濾後的黃瓜 Bounding Box
* **黃色 3D 骨架線**：經 3D 平面趨勢過濾與 Cubic Spline 擬合的最精確中心線
* **黃色文字**：最終估算之 3D 弧長（單位 `cm`）