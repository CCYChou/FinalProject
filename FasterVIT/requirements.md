# 環境需求說明文件

## Python版本要求
- Python >= 3.8

## 主要依賴套件

### 深度學習框架
```
torch>=2.0.0
torchvision>=0.15.0
```

### 數據處理與分析
```
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.0.0
```

### 視覺化工具
```
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.65.0
```

### 資料存取
```
pickle5>=0.0.11  # 用於讀取pickle文件
pathlib>=1.0.1   # 用於路徑處理
```

## 硬體需求

### GPU需求
- NVIDIA GPU with CUDA support
- 建議顯存 >= 8GB
- 支援CUDA 11.0或更高版本

### 系統需求
- RAM: 建議 >= 16GB
- 儲存空間: >= 10GB（用於模型和數據）

## 安裝說明

1. 創建虛擬環境
```bash
conda create -n wafer_env python=3.8
conda activate wafer_env
```

2. 安裝PyTorch（根據您的CUDA版本選擇適當的命令）
```bash
# CUDA 11.8
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. 安裝其他依賴
```bash
pip install -r requirements.txt
```

## 完整的requirements.txt內容
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.4.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.65.0
pickle5>=0.0.11
pathlib>=1.0.1
```

## 特殊說明

### CUDA相容性
- 確保NVIDIA驅動程序是最新的
- 確認PyTorch版本與CUDA版本相容
- 可以使用 `torch.cuda.is_available()` 檢查CUDA是否可用

### 資料格式需求
- 訓練數據需要以pickle格式儲存
- 圖像數據需要是32x32大小的灰度圖像
- 標籤需要是數值格式

### 模型相關
- FasterViT模型需要額外的配置檔案
- 預訓練權重將自動從Hugging Face下載

## 常見問題解決

1. CUDA相關錯誤
```bash
# 檢查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 檢查CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

2. 內存問題
- 如果遇到內存不足，可以減少batch_size
- 可以使用梯度累積來模擬更大的batch_size

3. 依賴衝突
- 建議使用虛擬環境避免依賴衝突
- 如果發生衝突，可以嘗試降級特定包的版本

## 維護與更新

- 定期檢查依賴套件的安全更新
- 確保與最新的CUDA版本相容
- 追蹤PyTorch的重要更新

## 聯絡支援

如果在安裝或運行過程中遇到問題，請：
1. 檢查錯誤日誌
2. 查閱文檔
3. 搜尋相關問題
4. 提交issue到專案儲存庫

## 版本歷史

### V1.0.0 (2024-01)
- 初始版本
- 支援CUDA 11.8
- 支援Python 3.8-3.10
