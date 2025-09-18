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
        self.data, self.labels = [], []
        self.classes = sorted(os.listdir(root_dir))
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if os.path.isdir(cls_dir):
                for fn in os.listdir(cls_dir):
                    path = os.path.join(cls_dir, fn)
                    if os.path.isfile(path):
                        self.data.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, lbl = self.data[idx], self.labels[idx]
        frames = self._load_video(path)
        if self.transform:
            frames = self.transform(frames)
        return frames, lbl

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        seq = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size)
            seq.append(frame)
            idx += 1
        cap.release()
        arr = np.array(seq, dtype=np.float32)
        N = 16
        if len(arr) > N:
            idxs = np.linspace(0, len(arr) - 1, N).astype(int)
            arr = arr[idxs]
        elif len(arr) < N:
            pad = np.zeros((N - len(arr), *self.target_size, 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        arr /= 255.0
        return arr.transpose(0, 3, 1, 2)

# ---------------------------
# 2. CNNEncoder
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.net(x)
        return x.view(B, T, -1)

# ---------------------------
# 3. Decoders
# ---------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, dim=256, heads=8, layers=2, classes=4):
        super().__init__()
        ld = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=256)
        self.dec = nn.TransformerDecoder(ld, num_layers=layers)
        self.fc = nn.Linear(dim, classes)

    def forward(self, vf, sf):
        v = vf.permute(1, 0, 2)
        s = sf.permute(1, 0, 2)
        out = self.dec(v, s).mean(dim=0)
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, dim=256, hid=512, layers=2, classes=4):
        super().__init__()
        self.lstm = nn.LSTM(dim * 2, hid, layers, batch_first=True)
        self.fc = nn.Linear(hid, classes)

    def forward(self, vf, sf):
        x = torch.cat([vf, sf], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class STGCNDecoder(nn.Module):
    def __init__(self, dim=256, classes=4):
        super().__init__()
        self.tc = nn.Conv2d(1, dim, (3, 1), padding=(1, 0))
        self.fc = nn.Linear(dim, classes)

    def forward(self, vf, sf):
        x = sf.unsqueeze(1)
        x = self.tc(x).mean(-1).mean(-1)
        v = vf.mean(dim=1)
        return self.fc(x + v)

# ---------------------------
# 4. CombinedDataset
# ---------------------------
class CombinedDataset(Dataset):
    def __init__(self, vd, sd, L=16):
        self.vd, self.sd, self.L = vd, sd, L

    def __len__(self):
        return min(len(self.vd), len(self.sd))

    def __getitem__(self, idx):
        v, lbl = self.vd[idx]
        s, lbl2 = self.sd[idx]
        assert lbl == lbl2
        return self._fix(v), self._fix(s), lbl

    def _fix(self, x):
        T = x.shape[0]
        if T > self.L:
            idxs = np.linspace(0, T - 1, self.L).astype(int)
            return x[idxs]
        elif T < self.L:
            pad = np.zeros((self.L - T, *x.shape[1:]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

# ---------------------------
# 5. Test & Plot
# ---------------------------
def test(model, dl, crit, dev):
    model.eval()
    total, correct1, correct5 = 0, 0, 0
    with torch.no_grad():
        for v, s, lbl in dl:
            v, s, lbl = v.to(dev), s.to(dev), lbl.to(dev)
            out = model(v, s)
            _, p1 = out.max(1)
            _, p5 = out.topk(5, 1)
            total += lbl.size(0)
            correct1 += p1.eq(lbl).sum().item()
            correct5 += p5.eq(lbl.view(-1, 1)).any(dim=1).sum().item()
    return 100. * correct1 / total, 100. * correct5 / total


def plot_accuracy(curves, names, title, path):
    plt.figure(figsize=(8, 6))
    for curve, name in zip(curves, names):
        plt.plot([1], curve, label=name)
    plt.xlabel('Run')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

# ---------------------------
# 6. Main (测试流程)
# ---------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据路径
    video_path = 'D:/Code_pytorch/zhongjie/pose detection/C3D-main/C3D-main/data/UCF-101'
    skl_path = 'D:/Code_pytorch/zhongjie/pose detection/C3D-main/ucf101_skl'

    vd = VideoDataset(video_path)
    sd = VideoDataset(skl_path)
    ds = CombinedDataset(vd, sd)
    dl = DataLoader(ds, batch_size=100, shuffle=False)

    # 模型集合
    models = {
        'Transformer': nn.Sequential(CNNEncoder(), TransformerDecoder()),
        'LSTM': nn.Sequential(CNNEncoder(), LSTMDecoder()),
        'STGCN': nn.Sequential(CNNEncoder(), STGCNDecoder())
    }
    for m in models.values():
        m.to(device)

    crit = nn.CrossEntropyLoss()
    results1, results5 = [], []
    names = list(models.keys())

    # 运行测试
    for name, model in models.items():
        acc1, acc5 = test(model, dl, crit, device)
        results1.append(acc1)
        results5.append(acc5)
        print(f'[{name}] Test Top-1: {acc1:.2f}%, Top-5: {acc5:.2f}%')

    # 绘制测试曲线
    plot_accuracy(results1, names, 'Test Top-1 Accuracy', 'test_top1.png')
    plot_accuracy(results5, names, 'Test Top-5 Accuracy', 'test_top5.png')
