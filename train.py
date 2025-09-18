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
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 1 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            idx += 1
        cap.release()
        frames = np.array(frames, dtype=np.float32)
        num_frames = 16
        if len(frames) > num_frames:
            idxs = np.linspace(0, len(frames)-1, num_frames).astype(int)
            frames = frames[idxs]
        elif len(frames) < num_frames:
            pad = np.zeros((num_frames-len(frames), *self.target_size, 3), dtype=np.float32)
            frames = np.concatenate((frames, pad), axis=0)
        frames /= 255.0
        return frames.transpose(0,3,1,2)  # (T, C, H, W)

# ---------------------------
# 2. CNNEncoder
# ---------------------------
class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        x = self.conv(x)
        x = x.view(B, T, -1)
        return x  # (B, T, feature_dim)

# ---------------------------
# 3. Decoders
# ---------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, num_classes=4):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=256)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, vf, sf):
        v = vf.permute(1,0,2)
        s = sf.permute(1,0,2)
        out = self.decoder(v, s)  # (T, B, D)
        out = out.mean(dim=0)     # (B, D)
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim*2, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, vf, sf):
        x = torch.cat([vf, sf], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out[:,-1])

class STGCNDecoder(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.temporal = nn.Conv2d(1, feature_dim, (3,1), padding=(1,0))
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, vf, sf):
        B, T, D = sf.size()
        x = sf.unsqueeze(1)            # (B,1,T,D)
        x = self.temporal(x)           # (B,F,T,D)
        x = x.mean(-1).mean(-1)        # (B,F)
        v = vf.mean(dim=1)             # (B,D)
        return self.fc(x + v)

# ---------------------------
# 4. Models
# ---------------------------
class CombinedModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.ve = CNNEncoder(feature_dim)
        self.se = CNNEncoder(feature_dim)
        self.decoder = TransformerDecoder(feature_dim, num_classes=num_classes)
    def forward(self, v, s):
        return self.decoder(self.ve(v), self.se(s))

class CombinedModelLSTM(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, num_layers=2, num_classes=4):
        super().__init__()
        self.ve = CNNEncoder(feature_dim)
        self.se = CNNEncoder(feature_dim)
        self.decoder = LSTMDecoder(feature_dim, hidden_dim, num_layers, num_classes)
    def forward(self, v, s):
        return self.decoder(self.ve(v), self.se(s))

class CombinedModelSTGCN(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.ve = CNNEncoder(feature_dim)
        self.se = CNNEncoder(feature_dim)
        self.decoder = STGCNDecoder(feature_dim, num_classes)
    def forward(self, v, s):
        return self.decoder(self.ve(v), self.se(s))

class RGBOnlyModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.ve = CNNEncoder(feature_dim)
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, v, _):
        x = self.ve(v).mean(dim=1)
        return self.fc(x)

class NoCrossAttentionModel(nn.Module):
    def __init__(self, feature_dim=256, num_classes=4):
        super().__init__()
        self.ve = CNNEncoder(feature_dim)
        self.se = CNNEncoder(feature_dim)
        self.fc = nn.Linear(feature_dim*2, num_classes)
    def forward(self, v, s):
        v = self.ve(v).mean(dim=1)
        s = self.se(s).mean(dim=1)
        return self.fc(torch.cat([v, s], dim=-1))

# ---------------------------
# 5. CombinedDataset
# ---------------------------
class CombinedDataset(Dataset):
    def __init__(self, vd, sd, seq_len=16):
        self.vd, self.sd, self.L = vd, sd, seq_len
    def __len__(self): return min(len(self.vd), len(self.sd))
    def __getitem__(self, i):
        v, l1 = self.vd[i]
        s, l2 = self.sd[i]
        assert l1==l2
        v = self._fix(v)
        s = self._fix(s)
        return (v, s), l1
    def _fix(self, x):
        T = x.shape[0]
        if T>self.L:
            return x[np.linspace(0,T-1,self.L).astype(int)]
        if T<self.L:
            pad = np.zeros((self.L-T, *x.shape[1:]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

# ---------------------------
# 6. Training & Plotting
# ---------------------------

def train(model, dl, crit, opt, dev):
    model.train()
    loss_sum, top1, top5, tot = 0.0, 0, 0, 0
    for (v, s), lbl in dl:
        v, s, lbl = v.to(dev), s.to(dev), lbl.to(dev)
        opt.zero_grad()
        out = model(v, s)
        loss = crit(out, lbl)
        loss.backward(); opt.step()
        loss_sum += loss.item()*lbl.size(0)
        tot += lbl.size(0)
        _, pred1 = out.max(1)
        top1 += pred1.eq(lbl).sum().item()
        _, pred5 = out.topk(5, 1)
        top5 += pred5.eq(lbl.view(-1,1)).any(1).sum().item()
    return loss_sum/tot, 100.*top1/tot, 100.*top5/tot


def plot_accuracy(accs, epochs, names, title, path):
    plt.figure(figsize=(8,6))
    for a,name in zip(accs, names):
        plt.plot(range(1, epochs+1), a, label=name)
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)')
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path)

# ---------------------------
# 7. Main
# ---------------------------
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_path = 'D:/Code_pytorch/.../UCF-101'
    skeleton_path = 'D:/Code_pytorch/.../ucf101_skl'

    vd = VideoDataset(video_path)
    sd = VideoDataset(skeleton_path)
    ds = CombinedDataset(vd, sd)
    loader = DataLoader(ds, batch_size=100, shuffle=True)

    models = {
        'RGBOnly': RGBOnlyModel(),
        'NoCrossAttn': NoCrossAttentionModel(),
        'Transformer': CombinedModel(),
        'LSTM': CombinedModelLSTM(),
        'STGCN': CombinedModelSTGCN()
    }
    for m in models.values(): m.to(device)

    optimizers = {n: optim.Adam(m.parameters(), lr=1e-4) for n,m in models.items()}
    criterion = nn.CrossEntropyLoss()
    history = {n:{'train_t1':[], 'train_t5':[]} for n in models}
    epochs = 50

    for epoch in range(1, epochs+1):
        print(f'Epoch [{epoch}/{epochs}]')
        for name, model in models.items():
            loss, t1, t5 = train(model, loader, criterion, optimizers[name], device)
            history[name]['train_t1'].append(t1)
            history[name]['train_t5'].append(t5)
            print(f'[{name}] Loss: {loss:.4f} | Top1: {t1:.2f}% | Top5: {t5:.2f}%')

    # Plot training curves
    names = list(models.keys())
    plot_accuracy([history[n]['train_t1'] for n in names], epochs, names, 'Train Top-1 Accuracy', 'train_top1.png')
    plot_accuracy([history[n]['train_t5'] for n in names], epochs, names, 'Train Top-5 Accuracy', 'train_top5.png')
