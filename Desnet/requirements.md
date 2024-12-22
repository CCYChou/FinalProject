# 專案環境需求

## 硬體需求
- GPU: NVIDIA GeForce RTX 4060 或更高
- RAM: 最小 16GB 建議
- Storage: SSD建議 (用於數據存儲和模型訓練)

## 作業系統
- Windows 10/11
- 或 Linux Ubuntu 20.04 LTS 或更高版本

## Python 環境
- Python 3.8 或更高版本
- Anaconda 或 Miniconda (建議使用虛擬環境)

## 主要依賴套件

### 深度學習框架
```
torch>=2.0.0
torchvision>=0.15.0
```

### 數據處理和分析
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

### 視覺化工具
```
matplotlib>=3.4.0
seaborn>=0.11.0
```

### 進度顯示
```
tqdm>=4.62.0
```

## 安裝步驟

1. 創建虛擬環境
```bash
conda create -n wafer-env python=3.8
conda activate wafer-env
```

2. 安裝 PyTorch (使用CUDA)
```bash
# 若使用 CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. 安裝其他依賴
```bash
pip install -r requirements.txt
```

## requirements.txt 內容
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

## CUDA 要求
- CUDA Toolkit 11.8 或相容版本
- cuDNN 8.0 或更高版本

## 數據需求
- 訓練數據集大小: 約 30,004 筆樣本
- 測試數據集大小: 約 7,502 筆樣本
- 儲存空間需求: 至少 10GB 可用空間 (包含數據集和模型)

## 開發工具建議
- IDE: PyCharm Professional 或 VS Code
- 版本控制: Git
- 數據版本控制: DVC (選用)

## 注意事項
1. GPU 記憶體需求
   - 訓練過程中至少需要 6GB GPU 記憶體
   - 批次大小(batch size)可根據GPU記憶體大小調整

2. 數據格式要求
   - 輸入圖像大小: 32x32 像素
   - 圖像格式: 單通道灰階圖像
   - 標籤格式: 9類分類標籤

3. 環境相容性
   - 確保所有套件版本相互兼容
   - 建議使用虛擬環境以避免套件衝突

4. 性能考量
   - 建議使用 SSD 以提高數據讀取速度
   - 可考慮使用數據預載入以提高訓練效率

## 安裝驗證
執行以下指令確認環境安裝正確：

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 故障排除
1. CUDA 相關問題
   - 確認 NVIDIA 驅動程式是否最新
   - 確認 CUDA Toolkit 版本與 PyTorch 相容

2. 記憶體問題
   - 減小批次大小
   - 使用梯度累積
   - 考慮使用混合精度訓練

3. 數據載入問題
   - 確認數據路徑正確
   - 檢查數據格式是否符合要求
   - 確認磁碟空間充足