"""Baseline 方法。

- ProtoNet：Prototypical Network（类均值原型 + 余弦最近邻），few-shot 标准参考。
- CELinearProbe：在 support 特征上训练线性分类头 + 交叉熵（强基线：label smoothing + weight decay）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet:
    def __init__(self, n_way: int):
        self.n_way = n_way
        self.proto = None

    def fit(self, support_feats, support_labels):
        support_feats = np.asarray(support_feats, dtype=np.float32)
        support_labels = np.asarray(support_labels)
        self.proto = np.stack([
            support_feats[support_labels == c].mean(0) for c in range(self.n_way)
        ])
        self.proto = self.proto / (np.linalg.norm(self.proto, axis=1, keepdims=True) + 1e-12)
        return self

    def predict(self, query_feats):
        return self.predict_proba(query_feats).argmax(1)

    def predict_proba(self, query_feats, temperature=1.0):
        sim = np.asarray(query_feats, dtype=np.float32) @ self.proto.T
        logits = sim / temperature
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)


class CELinearProbe:
    def __init__(self, feat_dim: int, n_way: int, lr: float = 0.1, epochs: int = 200,
                 device: str = "cpu", seed: int = 0, weight_decay: float = 1e-4,
                 label_smoothing: float = 0.1):
        self.feat_dim = feat_dim
        self.n_way = n_way
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.head = None

    def fit(self, support_feats, support_labels):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(support_feats, dtype=torch.float32)
        y = torch.as_tensor(support_labels, dtype=torch.long)
        self.head = nn.Linear(self.feat_dim, self.n_way).to(self.device)
        opt = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Xd, yd = X.to(self.device), y.to(self.device)
        self.head.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.head(Xd), yd, label_smoothing=self.label_smoothing)
            loss.backward()
            opt.step()
        self.head.eval()
        return self

    def logits(self, query_feats):
        X = torch.as_tensor(query_feats, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.head(X).cpu().numpy()

    def predict(self, query_feats):
        return self.logits(query_feats).argmax(1)

    def predict_proba(self, query_feats):
        from src.metrics import softmax
        return softmax(self.logits(query_feats))
