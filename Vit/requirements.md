# 環境需求清單

## Python 版本
- Python >= 3.8

## 主要套件需求

### 深度學習框架
- torch==2.1.0  # PyTorch深度學習框架
- einops==0.7.0  # 張量操作工具

### 數據處理
- numpy==1.24.0  # 數值計算庫
- pandas==2.1.0  # 數據分析庫
- scikit-learn==1.3.0  # 機器學習工具包

### 視覺化工具
- matplotlib==3.7.1  # 繪圖庫
- seaborn==0.12.2  # 統計資料視覺化庫

### 進度顯示
- tqdm==4.65.0  # 進度條顯示

## 硬體需求
- NVIDIA GPU (建議 >= 8GB VRAM)
- CUDA Toolkit >= 11.8 (配合PyTorch版本)
- RAM >= 16GB

## 安裝指南

### 使用 pip 安裝
```bash
pip install -r requirements.txt
```

### 使用 conda 創建環境
```bash
conda create -n wafer-env python=3.8
conda activate wafer-env
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
pip install -r requirements.txt
```

## 注意事項
1. GPU版本的PyTorch安裝指令可能因為不同的CUDA版本而異，請參考[PyTorch官網](https://pytorch.org/)的安裝指南。
2. 建議使用虛擬環境來安裝套件，避免與其他專案的套件發生衝突。
3. 若使用CPU訓練，可以安裝CPU版本的PyTorch，但訓練時間會大幅增加。

## 套件版本更新說明
- 上述版本號為開發時使用的版本，較新的版本也可能相容。
- 若遇到相容性問題，建議先使用列出的版本號。

## 系統相容性
- 作業系統：Windows 10/11, Linux (Ubuntu 20.04+), macOS
- 需要安裝NVIDIA驅動程式（若使用GPU）
- 建議使用NVIDIA RTX系列顯示卡以獲得最佳性能

## 額外工具建議
- VSCode 或 PyCharm 作為IDE
- Git 進行版本控制
- NVIDIA-SMI 監控GPU使用狀況

## 常見問題解決
1. CUDA相關錯誤：
   - 確認NVIDIA驅動程式是否正確安裝
   - 確認CUDA版本與PyTorch版本相容
   
2. 記憶體不足：
   - 減少batch size
   - 使用梯度累積
   - 使用混合精度訓練

3. 套件衝突：
   - 建立新的虛擬環境
   - 按照順序安裝套件
   - 檢查套件相依性

## 開發工具建議
- Jupyter Notebook/Lab：用於互動式開發和實驗
- Tensorboard：用於訓練過程視覺化
- Docker：用於環境封裝和部署（選用）