import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm

import gc
import os
import time
import random
from datetime import datetime

from PIL import Image
from tqdm import tqdm
from sklearn import model_selection, metrics



def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1001)

# 配置参数
data_path = r"data/cifar-10"  #炮声都学习，都应该用
MODEL_PATH = r"data/cifar-10/jx_vit_base_p16_224-80ecf9dd.pth"

IMG_SIZE = 224
BATCH_SIZE = 8  # CPU需要更小的batch size
LR = 1e-05  # 使用更小的学习率
N_EPOCHS = 5  # 减少训练轮数

# 数据准备
df = pd.read_csv(r"data/cifar-10/trainLabels.csv")
df['id'] = df['id'].astype(str)
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.1, random_state=42, stratify=df.label.values
)


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path=data_path, mode="train", transforms=None):
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.data_dir = "train_images" if mode == "train" else "test_images"

        # 定义类别名称到整数的映射
        self.class_to_idx = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label_str = self.df_data[index]  # 假设列顺序是 [文件名, 类别名]
        img_name = str(img_name)+".png"
        img_path = os.path.join(self.data_path, self.data_dir, img_name)

        # 将类别名称转换为整数标签
        label = self.class_to_idx[label_str]

        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(img)
        return image, label


# 数据增强（简化版）
transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transforms_valid = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# 模型定义（使用更小的模型）
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        if pretrained:
            self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)






# 训练函数（优化内存使用）
def train_cpu(model, train_loader, valid_loader, criterion, optimizer):
    best_valid_loss = float('inf')

    for epoch in range(1, N_EPOCHS + 1):
        print(f'Epoch {epoch}/{N_EPOCHS}')

        # 训练阶段
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in tqdm(train_loader, desc='Training'):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # 直接在CPU上运行
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(1) == labels).float().mean()
            train_loss += loss.item() * images.size(0)
            train_acc += acc.item() * images.size(0)

        # 验证阶段
        valid_loss, valid_acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc='Validating'):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(1) == labels).float().mean()
                valid_loss += loss.item() * images.size(0)
                valid_acc += acc.item() * images.size(0)

        # 计算指标
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_acc / len(valid_loader.dataset)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model_gpu.pth')
            print('Model Saved!')

    return model





if __name__ == '__main__':
    # 准备数据
    train_dataset = CassavaDataset(train_df, transforms=transforms_train)
    valid_dataset = CassavaDataset(valid_df, transforms=transforms_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # 减少工作线程数
        pin_memory=False  # 关闭内存锁页
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # 初始化模型
    model = ViTBase16(n_classes=10, pretrained=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 开始训练
    print("Starting training on GPU...")
    start_time = time.time()
    trained_model = train_cpu(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


