import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from model.__init__ import FasterViT  # Import the FasterViT model
from pathlib import Path
import os

class WaferDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data).unsqueeze(1)  # [B, 1, 32, 32]
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # 使用bicubic插值將圖像從32x32調整為224x224
        x = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)(x)
        
        # 將單通道圖像轉換為三通道並進行標準化
        x = x.repeat(3, 1, 1)  # [3, 224, 224]
        
        # 根據FasterViT的預訓練要求進行標準化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        x = normalize(x)

        if self.transform:
            x = self.transform(x)

        return x, y

class WaferModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(WaferModel, self).__init__()
        # 初始化 FasterViT 模型
        self.model = FasterViT(
            dim=64,  # FasterViT-0的基礎維度
            in_dim=64,
            depths=[2, 3, 6, 5],  # FasterViT-0的層深度配置
            num_heads=[2, 4, 8, 16],  # 注意力頭數配置
            window_size=[7, 7, 7, 7],
            ct_size=2,
            mlp_ratio=4,
            resolution=224,
            drop_path_rate=0.2,
            in_chans=3,
            num_classes=num_classes,
            hat=[False, False, True, False]
        )
        
        if pretrained:
            # 載入預訓練權重
            url = 'https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar'
            model_dir = Path('./pretrained_models')
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / 'faster_vit_0.pth.tar'
            
            if not model_path.exists():
                print(f"Downloading pretrained model from {url}")
                torch.hub.download_url_to_file(url=url, dst=str(model_path))
            
            try:
                checkpoint = torch.load(str(model_path))
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print("Successfully loaded pretrained weights")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")
                print("Training from scratch")

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    """訓練模型"""
    model = model.to(device)
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    best_epoch = 0

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 更新學習率
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # 驗證階段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_wafer_model.pth')

    print(f'\nBest model was saved at epoch {best_epoch+1} with validation accuracy: {best_val_acc:.2f}%')
    return train_losses, val_accuracies

def main():
    # 檢查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 設定隨機種子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 讀取數據
    print("正在讀取數據...")
    data_path = './data/'

    with open(data_path + 'x_train_org_20210614.pickle', 'rb') as data:
        x_train = pickle.load(data)
    with open(data_path + 'y_train_org_20210614.pickle', 'rb') as data:
        y_train = pickle.load(data)
    with open(data_path + 'x_test_20210614.pickle', 'rb') as data:
        x_test = pickle.load(data)
    with open(data_path + 'y_test_20210614.pickle', 'rb') as data:
        y_test = pickle.load(data)

    print(f"訓練集大小: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"測試集大小: x_test: {x_test.shape}, y_test: {y_test.shape}")

    # 標籤編碼
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train.ravel())
    y_test = label_encoder.transform(y_test.ravel())

    # 創建數據集和數據加載器
    train_dataset = WaferDataset(x_train, y_train)
    test_dataset = WaferDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 初始化模型
    num_classes = len(np.unique(y_train))
    print(f"\n類別數量: {num_classes}")
    model = WaferModel(num_classes=num_classes, pretrained=True)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 訓練模型
    train_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler,
        num_epochs=50, device=device
    )

    # 繪製訓練過程
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # 評估模型
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'\nTest Accuracy: {100 * correct / total:.2f}%')

    # 繪製混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 輸出分類報告
    print('\nClassification Report:')
    class_names = [f'Class {i}' for i in range(num_classes)]
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    main()
