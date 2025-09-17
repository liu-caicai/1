import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 1. VideoDataset
# ---------------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(128, 128)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for video_name in os.listdir(class_dir):
                    video_path = os.path.join(class_dir, video_name)
                    if os.path.isfile(video_path):
                        self.data.append(video_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        label = self.labels[idx]
        frames = self._load_video(video_path)
        if self.transform:
            frames = self.transform(frames)
        return frames, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_skip = 1
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            frame_idx += 1

        cap.release()

        frames = np.array(frames, dtype=np.float32)
        num_frames = 16  
        if len(frames) > num_frames:
            idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
            frames = frames[idx]
        elif len(frames) < num_frames:
            padding = np.zeros((num_frames - len(frames), *self.target_size, 3), dtype=np.float32)
            frames = np.concatenate((frames, padding), axis=0)

        frames /= 255.0
        # (T, H, W, C) => (T, C, H, W)
        return frames.transpose(0, 3, 1, 2)


# ---------------------------
# 2. CNNEncoder
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        batch_size, time_steps, c, h, w = x.size()
        # 先把时序维度合并到 batch 维度
        x = x.view(batch_size * time_steps, c, h, w)
        x = self.conv(x)
        # 再 reshape 回来 (batch_size, time_steps, 特征维度)
        x = x.view(batch_size, time_steps, -1)
        return x


# ---------------------------
# 3. TransformerDecoder
# ---------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, num_layers=2, num_classes=4):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=256
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, video_features, skeleton_features):
        # Transformer: (T, B, D) 格式
        video_features = video_features.permute(1, 0, 2)     # (B, T, D) -> (T, B, D)
        skeleton_features = skeleton_features.permute(1, 0, 2)
        fused_features = self.decoder(video_features, skeleton_features)
        # 融合后取时序平均
        fused_features = fused_features.mean(dim=0)
        out = self.fc(fused_features)
        return out


# ---------------------------
# 4. CombinedModel
# ---------------------------
class CombinedModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super(CombinedModel, self).__init__()
        self.video_encoder = CNNEncoder(feature_dim)
        self.skeleton_encoder = CNNEncoder(feature_dim)
        self.decoder = TransformerDecoder(feature_dim, num_classes=num_classes)

    def forward(self, video, skeleton):
        video_features = self.video_encoder(video)
        skeleton_features = self.skeleton_encoder(skeleton)
        out = self.decoder(video_features, skeleton_features)
        return out


# ---------------------------
# 5. RGBOnlyModel
# ---------------------------
class RGBOnlyModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super(RGBOnlyModel, self).__init__()
        self.video_encoder = CNNEncoder(feature_dim)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, video, _):
        video_features = self.video_encoder(video)  # (B, T, D)
        video_features = video_features.mean(dim=1) # (B, D)
        out = self.fc(video_features)
        return out


# ---------------------------
# 6. NoCrossAttentionModel
# ---------------------------
class NoCrossAttentionModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=5):
        super(NoCrossAttentionModel, self).__init__()
        self.video_encoder = CNNEncoder(feature_dim)
        self.skeleton_encoder = CNNEncoder(feature_dim)
        self.fc = nn.Linear(2 * feature_dim, num_classes)

    def forward(self, video, skeleton):
        video_features = self.video_encoder(video).mean(dim=1)
        skeleton_features = self.skeleton_encoder(skeleton).mean(dim=1)
        fused_features = torch.cat([video_features, skeleton_features], dim=-1)
        out = self.fc(fused_features)
        return out


# ---------------------------
# 7. CombinedDataset (含 DTW + 二次固定长度)
# ---------------------------
class CombinedDataset(Dataset):
    def __init__(self, video_dataset, skeleton_dataset, final_seq_len=16):
        """
        :param final_seq_len: 对齐结束后，统一再裁剪或补零到的最终帧数
        """
        self.video_dataset = video_dataset
        self.skeleton_dataset = skeleton_dataset
        self.final_seq_len = final_seq_len

    def __len__(self):
        return min(len(self.video_dataset), len(self.skeleton_dataset))

    def __getitem__(self, idx):
        video_data, video_label = self.video_dataset[idx]
        skeleton_data, skeleton_label = self.skeleton_dataset[idx]
        assert video_label == skeleton_label, "Labels of video and skeleton must match"

        video_data_aligned, skeleton_data_aligned = self._align_modalities_dtw(
            video_data, skeleton_data
        )

        # 再次对长度进行统一，例如都处理到 self.final_seq_len 帧
        video_data_aligned = self._fix_length(video_data_aligned, self.final_seq_len)
        skeleton_data_aligned = self._fix_length(skeleton_data_aligned, self.final_seq_len)

        return (video_data_aligned, skeleton_data_aligned), video_label

    def _fix_length(self, data, final_len):
        """
        将 data 的时序维度变为 final_len
        data: shape (T, C, H, W)
        可以自行选择采样或补零，这里以补零示例
        """
        T, C, H, W = data.shape
        if T == final_len:
            return data
        elif T > final_len:
            # 截断（线性间隔采样）
            indices = np.linspace(0, T - 1, final_len).astype(int)
            data = data[indices]
            return data
        else:
            # 补零
            pad = np.zeros((final_len - T, C, H, W), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
            return data

    def _compute_frame_distance(self, frame1, frame2):
        """
        示例：两帧均值向量间欧几里得距离
        frame: (C, H, W)
        """
        mean1 = frame1.mean(axis=(1, 2))  # shape (C,)
        mean2 = frame2.mean(axis=(1, 2))  # shape (C,)
        return np.linalg.norm(mean1 - mean2)

    def _dtw(self, seq1, seq2):
        """
        经典 DTW 实现
        seq1, seq2 shape: (T, C, H, W)
        """
        T1 = seq1.shape[0]
        T2 = seq2.shape[0]

        # 计算 cost 矩阵
        cost_matrix = np.zeros((T1, T2))
        for i in range(T1):
            for j in range(T2):
                cost_matrix[i, j] = self._compute_frame_distance(seq1[i], seq2[j])

        # 动态规划矩阵
        dp = np.full((T1 + 1, T2 + 1), np.inf)
        dp[0, 0] = 0.0

        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                cost = cost_matrix[i - 1, j - 1]
                dp[i, j] = cost + min(
                    dp[i - 1, j],
                    dp[i, j - 1],
                    dp[i - 1, j - 1]
                )

        # 回溯获取路径
        path = []
        i, j = T1, T2
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            neighbors = [
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            ]
            argmin = np.argmin(neighbors)
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

        path.reverse()
        return path

    def _align_modalities_dtw(self, seq1, seq2):
        """
        返回对齐后的序列
        seq1, seq2: (T, C, H, W)
        """
        path = self._dtw(seq1, seq2)
        aligned_seq1 = []
        aligned_seq2 = []
        for (i, j) in path:
            aligned_seq1.append(seq1[i])
            aligned_seq2.append(seq2[j])

        aligned_seq1 = np.stack(aligned_seq1, axis=0)
        aligned_seq2 = np.stack(aligned_seq2, axis=0)
        return aligned_seq1, aligned_seq2


# ---------------------------
# 7+. CombinedDatasetNoAlignment (去除模态对齐)
# ---------------------------
class CombinedDatasetNoAlignment(Dataset):
    """
    与 CombinedDataset 类似，但去除了 DTW 对齐的步骤，仅仅把对应 index 的 video 与 skeleton 拼在一起，
    并做一次最终长度的 fix 操作(保持与 CombinedDataset 相同的 final_seq_len)。
    """
    def __init__(self, video_dataset, skeleton_dataset, final_seq_len=16):
        self.video_dataset = video_dataset
        self.skeleton_dataset = skeleton_dataset
        self.final_seq_len = final_seq_len

    def __len__(self):
        return min(len(self.video_dataset), len(self.skeleton_dataset))

    def __getitem__(self, idx):
        video_data, video_label = self.video_dataset[idx]
        skeleton_data, skeleton_label = self.skeleton_dataset[idx]
        assert video_label == skeleton_label, "Labels of video and skeleton must match"

        # 不进行对齐，直接做 fix_length
        video_data_fixed = self._fix_length(video_data, self.final_seq_len)
        skeleton_data_fixed = self._fix_length(skeleton_data, self.final_seq_len)

        return (video_data_fixed, skeleton_data_fixed), video_label

    def _fix_length(self, data, final_len):
        """
        不做对齐，仅仅保证长度一致。与上面相同的实现即可。
        """
        T, C, H, W = data.shape
        if T == final_len:
            return data
        elif T > final_len:
            indices = np.linspace(0, T - 1, final_len).astype(int)
            data = data[indices]
            return data
        else:
            pad = np.zeros((final_len - T, C, H, W), dtype=data.dtype)
            data = np.concatenate([data, pad], axis=0)
            return data


# ---------------------------
# 8. train, test, plot_accuracy
# ---------------------------
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        (video, skeleton), labels = batch
        video, skeleton, labels = video.to(device), skeleton.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(video, skeleton)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return running_loss / len(dataloader), accuracy


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            (video, skeleton), labels = batch
            video, skeleton, labels = video.to(device), skeleton.to(device), labels.to(device)

            outputs = model(video, skeleton)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return running_loss / len(dataloader), accuracy


def plot_accuracy(accuracies, epochs, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    for i, acc in enumerate(accuracies):
        plt.plot(range(1, epochs + 1), acc, label=labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ---------------------------
# 9. 主程序
# ---------------------------
if __name__ == "__main__":
    # 超参数
    num_classes = 3
    batch_size = 100       # 调小 batch_size 方便测试
    epochs = 50
    lr = 1e-4

    # 数据路径
    video_data_path = "D:/Code_pytorch/zhongjie/pose detection/C3D-main/C3D-main/data/UCF-101"
    skeleton_data_path = "D:/Code_pytorch/zhongjie/pose detection/C3D-main/ucf101_skl"

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集与 DataLoader
    video_dataset = VideoDataset(video_data_path)
    skeleton_dataset = VideoDataset(skeleton_data_path)

    # 1) 使用含对齐的 CombinedDataset
    combined_dataset = CombinedDataset(video_dataset, skeleton_dataset, final_seq_len=16)
    combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # 2) 使用无对齐的 CombinedDatasetNoAlignment（消融实验）
    combined_noalign_dataset = CombinedDatasetNoAlignment(video_dataset, skeleton_dataset, final_seq_len=16)
    combined_noalign_dataloader = DataLoader(combined_noalign_dataset, batch_size=batch_size, shuffle=True)

    # 模型初始化
    rgb_only_model = RGBOnlyModel(num_classes=num_classes).to(device)
    no_cross_attention_model = NoCrossAttentionModel(num_classes=num_classes).to(device)
    combined_model = CombinedModel(num_classes=num_classes).to(device)
    # 为消融实验（无对齐）也准备一个相同模型
    combined_model_noalign = CombinedModel(num_classes=num_classes).to(device)

    # 优化器
    rgb_optimizer = optim.Adam(rgb_only_model.parameters(), lr=lr)
    no_cross_attention_optimizer = optim.Adam(no_cross_attention_model.parameters(), lr=lr)
    combined_optimizer = optim.Adam(combined_model.parameters(), lr=lr)
    combined_noalign_optimizer = optim.Adam(combined_model_noalign.parameters(), lr=lr)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 记录训练与测试精度
    rgb_train_accuracies, rgb_test_accuracies = [], []
    no_cross_attention_train_accuracies, no_cross_attention_test_accuracies = [], []
    combined_train_accuracies, combined_test_accuracies = [], []
    noalign_train_accuracies, noalign_test_accuracies = [], []

    for epoch in range(epochs):
        print(f"========== Epoch {epoch+1}/{epochs} ==========")

        # 1) 训练 RGB Only model
        train_loss, train_acc = train(rgb_only_model, combined_dataloader, criterion, rgb_optimizer, device)
        test_loss, test_acc = test(rgb_only_model, combined_dataloader, criterion, device)
        rgb_train_accuracies.append(train_acc)
        rgb_test_accuracies.append(test_acc)
        print(f"[RGB Only] => Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # 2) 训练 No Cross Attention
        train_loss, train_acc = train(no_cross_attention_model, combined_dataloader, criterion, no_cross_attention_optimizer, device)
        test_loss, test_acc = test(no_cross_attention_model, combined_dataloader, criterion, device)
        no_cross_attention_train_accuracies.append(train_acc)
        no_cross_attention_test_accuracies.append(test_acc)
        print(f"[No Cross Attention] => Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # 3) 训练 Combined Model（含对齐）
        train_loss, train_acc = train(combined_model, combined_dataloader, criterion, combined_optimizer, device)
        test_loss, test_acc = test(combined_model, combined_dataloader, criterion, device)
        combined_train_accuracies.append(train_acc)
        combined_test_accuracies.append(test_acc)
        print(f"[Combined Model] => Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # 4) 训练 Combined Model（**去除对齐**）
        train_loss, train_acc = train(combined_model_noalign, combined_noalign_dataloader, criterion, combined_noalign_optimizer, device)
        test_loss, test_acc = test(combined_model_noalign, combined_noalign_dataloader, criterion, device)
        noalign_train_accuracies.append(train_acc)
        noalign_test_accuracies.append(test_acc)
        print(f"[No Aligned Model] => Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # 绘制训练精度
    plot_accuracy(
        [
            rgb_train_accuracies,
            no_cross_attention_train_accuracies,
            combined_train_accuracies,
            noalign_train_accuracies
        ],
        epochs,
        ["RGB Only", "NoCrossAttn", "Combined", "NoAlign"],
        title="Train Accuracy",
        save_path="train_accuracy.png"
    )

    # 绘制测试精度
    plot_accuracy(
        [
            rgb_test_accuracies,
            no_cross_attention_test_accuracies,
            combined_test_accuracies,
            noalign_test_accuracies
        ],
        epochs,
        ["RGB Only", "NoCrossAttn", "Combined", "NoAlign"],
        title="Test Accuracy",
        save_path="test_accuracy.png"
    )
