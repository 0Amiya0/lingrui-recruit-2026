"""评估指标：Accuracy / Macro-F1 / ECE / 噪声识别准确率。"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=axis, keepdims=True)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def temperature_scale(logits, labels, n_iters=200, lr=0.1, seed=0):
    """在给定 (logits, labels) 上拟合 temperature，返回标量 T（用于校准 ECE）。"""
    torch.manual_seed(seed)
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.Adam([T], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            T.clamp_(min=1e-3)
    return float(T.detach())


def expected_calibration_error(probs, y_true, n_bins=15):
    """ECE：按预测置信度分桶，|桶内准确率 - 桶内平均置信度| 的加权平均。"""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize 到 [0, n_bins-1]，边界用 bins[1:-1]
    bin_ids = np.digitize(conf, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def noise_detection_metrics(pred_noise_mask, true_noise_mask):
    """噪声样本识别准确率：预测为噪声的 mask vs 真实噪声 mask。

    pred_noise_mask: 模型判定为「噪声」的样本（True=判为噪声）
    true_noise_mask: 真实被污染的样本（True=确实是噪声）
    返回 precision / recall / f1（召回=找出多少真噪声，精确=判为噪声里有多少是真的）。
    """
    pred = np.asarray(pred_noise_mask, dtype=bool)
    true = np.asarray(true_noise_mask, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
