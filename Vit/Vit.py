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
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = attn @ v
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, 
                 num_layers, num_classes, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.ln(x)
        x = x[:, 0]  # 只使用 [CLS] token
        x = self.head(x)
        return x

class WaferDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, device='cuda'):
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

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * correct / total

        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_wafer_vit_model.pth')

    print(f'\nBest model was saved at epoch {best_epoch+1} with validation accuracy: {best_val_acc:.2f}%')
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, device, label_encoder=None):
    """評估模型"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print("\n最終評估結果：")
    print(f"準確率: {accuracy:.2f}%")
    print(f"正確預測數: {correct}")
    print(f"總樣本數: {total}")

    if label_encoder is not None:
        print("\n各類別統計：")
        for idx, label in enumerate(label_encoder.classes_):
            mask = np.array(all_labels) == idx
            if np.any(mask):
                class_correct = (np.array(all_predictions)[mask] == idx).sum()
                class_total = mask.sum()
                print(f"{label}: {class_correct}/{class_total} ({100.0 * class_correct / class_total:.2f}%)")
    
    # 顯示混淆矩陣
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 顯示分類報告
    print("\n分類報告：")
    print(classification_report(all_labels, all_predictions, 
                              target_names=label_encoder.classes_ if label_encoder else None))

def main():
    # 檢查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 設定隨機種子以確保結果可重現
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 讀取數據
    print("正在讀取數據...")
    data_path = './data/'  # 請修改為你的實際數據路徑

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

    print("\n標籤映射:")
    for label, idx in enumerate(label_encoder.classes_):
        print(f"- {idx} -> {label}")

    # 創建數據集和數據加載器
    train_dataset = WaferDataset(x_train, y_train)
    test_dataset = WaferDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ViT 參數設定
    image_size = x_train.shape[1]  # 假設圖片是正方形
    patch_size = 16  # 可以根據圖片大小調整
    in_channels = 1
    embed_dim = 256
    num_heads = 8
    num_layers = 6
    num_classes = len(np.unique(y_train))

    # 初始化 ViT 模型
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # ViT通常使用較小的學習率

    # 訓練模型
    train_losses, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=50, device=device
    )

    # 繪製訓練過程
    plt.figure(figsize=(15, 5))

    # 損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 準確率曲線
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_vit.png')
    plt.show()

    # 顯示訓練結果統計
    print("\n訓練結果統計：")
    print(f"最終訓練損失: {train_losses[-1]:.4f}")
    print(f"最終驗證準確率: {val_accuracies[-1]:.2f}%")

    best_epoch = np.argmax(val_accuracies)
    print(f"\n最佳結果：")
    print(f"最佳驗證準確率: {val_accuracies[best_epoch]:.2f}%")
    print(f"在第 {best_epoch + 1} 個epoch")
    print(f"對應的訓練損失: {train_losses[best_epoch]:.4f}")

    # 載入最佳模型並進行評估
    model.load_state_dict(torch.load('best_wafer_vit_model.pth'))
    evaluate_model(model, test_loader, device, label_encoder)

    return train_losses, val_accuracies

if __name__ == '__main__':
    train_losses, val_accuracies = main()
