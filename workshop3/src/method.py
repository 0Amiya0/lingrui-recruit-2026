"""核心方法：距离驱动样本筛选 + 鲁棒损失（GCE / SCE）。

思想：把经典「小损失样本 = 干净样本」迁移到 few-shot 原型空间，变成
「大距离样本 = 疑似噪声」。筛选后再用鲁棒损失训练线性分类头，抵御残余噪声。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics import softmax


# ---------- 鲁棒损失 ----------
def gce_loss(logits, targets, q=0.7):
    """广义交叉熵：L = (1 - p_y^q) / q，q 越小越鲁棒。"""
    p = F.softmax(logits, dim=1)
    p_y = p.gather(1, targets.view(-1, 1)).squeeze(1)
    return ((1 - p_y ** q) / q).mean()


def sce_loss(logits, targets, alpha=1.0, beta=0.1, eps=1e-4):
    """对称交叉熵：CE + beta * RCE，惩罚对错误类别的过度自信。"""
    ce = F.cross_entropy(logits, targets)
    p = F.softmax(logits, dim=1)
    C = logits.size(1)
    q = torch.full_like(p, eps)
    q.scatter_(1, targets.view(-1, 1), 1.0 - eps)
    rce = -(p * q.log()).sum(1).mean()
    return alpha * ce + beta * rce


def get_loss(loss_name, **kw):
    if loss_name == "ce":
        return lambda logits, targets: F.cross_entropy(
            logits, targets, label_smoothing=kw.get("label_smoothing", 0.0))
    if loss_name == "gce":
        return lambda logits, targets: gce_loss(logits, targets, q=kw.get("q", 0.7))
    if loss_name == "sce":
        return lambda logits, targets: sce_loss(
            logits, targets, alpha=kw.get("alpha", 1.0), beta=kw.get("beta", 0.1))
    raise ValueError(f"未知损失: {loss_name}")


# ---------- 距离驱动样本筛选 ----------
def distance_based_selection(support_feats, support_labels, n_way, n_iters=5, beta=1.0, top_tau=None):
    """迭代 trimmed-mean 原型，把「大距离」样本判为噪声。

    返回 (pred_noise_mask, keep_mask, dist_scores)：pred_noise_mask=True 表示被判为噪声；
    dist_scores 为最终每个样本到其标注类原型的余弦距离（越大越可能是噪声，用于 AUROC）。
    """
    feats = np.asarray(support_feats, dtype=np.float32)
    labels = np.asarray(support_labels)
    n = len(feats)
    keep = np.ones(n, dtype=bool)
    d = np.zeros(n, dtype=np.float32)   # 到标注类原型的余弦距离（噪声似然分数）
    proto = np.zeros((n_way, feats.shape[1]), dtype=np.float32)

    for _ in range(n_iters):
        for c in range(n_way):
            cls_mask = labels == c
            sel = feats[keep & cls_mask]
            if len(sel) == 0:
                sel = feats[cls_mask]  # 回退到该类全部样本，避免空原型
            proto[c] = sel.mean(0)
        proto_norm = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - (feats * proto_norm[labels]).sum(1)  # 到标注类原型的余弦距离

        if top_tau is not None:
            thr = np.percentile(d, 100 * (1 - top_tau))
        else:
            thr = d.mean() + beta * d.std()

        new_keep = d <= thr
        # 保证每类至少保留一个样本，且没有过度剔除才更新
        if new_keep.sum() >= n_way and not (new_keep == keep).all():
            keep = new_keep
        else:
            break
    return ~keep, keep, d


# ---------- 线性分类头训练 ----------
def train_linear_head(feats, labels, feat_dim, n_way, loss_name="ce", loss_kw=None,
                      lr=0.1, epochs=200, device="cpu", seed=0,
                      weight_decay=1e-4, label_smoothing=0.1):
    loss_kw = loss_kw or {}
    loss_kw.setdefault("label_smoothing", label_smoothing)
    torch.manual_seed(seed)
    loss_fn = get_loss(loss_name, **loss_kw)
    X = torch.as_tensor(feats, dtype=torch.float32).to(device)
    y = torch.as_tensor(labels, dtype=torch.long).to(device)
    head = nn.Linear(feat_dim, n_way).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(X), y)
        loss.backward()
        opt.step()
    head.eval()
    return head


def head_proba(head, query_feats, device):
    X = torch.as_tensor(query_feats, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = head(X).cpu().numpy()
    return softmax(logits)
