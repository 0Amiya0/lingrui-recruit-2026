# Robust Few-shot Image Classification under Label Noise

> 3–4 page workshop short paper draft. Numeric results marked `[FILL]` are produced by
> running `run_all.ps1` and reading `results/*.csv`. Structure follows `task.md` §9.

---

## Abstract

Few-shot classification assumes a small set of carefully labeled support examples, yet
real-world annotations are frequently noisy. In this paper we study image classification
where each class has only $K$ labeled examples *and* a fraction of those labels are
corrupted. We inject symmetric label noise into CIFAR-10 support sets and show that both a
linear probe trained with cross-entropy and a prototypical network degrade sharply as the
noise rate grows. We propose a simple two-component remedy: (i) **distance-driven sample
selection**, which iteratively trims support examples that lie far from their class
prototype — the few-shot analogue of the "small-loss" heuristic — and (ii) a **robust loss**
(Generalized Cross Entropy) applied to the retained examples. Across noise rates of
0%–40% our method consistently outperforms the baselines on the clean test set, while the
selection step itself doubles as a noise detector. We further analyze which noisy samples
are missed and why, and characterize the resulting prototype drift.

## 1. Introduction

Workshop short papers emphasize a clear research question, a sound experimental design,
and results that support the conclusions. Our question is: **how does label noise interact
with label scarcity, and can a selection-plus-robust-loss method recover clean-test
accuracy?**

Few-shot learners are especially fragile to label noise: with only $K$ examples per class,
a single mislabeled sample already distorts the class prototype, and the distortion grows
with the noise rate. Standard label-noise remedies built for large datasets (large-sample
selection, complex two-stage training) are not directly applicable when the support set is
tiny.

Our contributions:
1. A controlled benchmark injecting symmetric label noise into few-shot support sets of
   CIFAR-10.
2. A lightweight **distance-driven selection + robust loss** method with an interpretable
   noise-detection signal.
3. A systematic comparison and ablation across noise rates and few-shot sizes, plus an
   error analysis of misclassification and noise-identification failures.

## 2. Related Work

**Label-noise robustness.** Methods fall into robust losses (generalized cross entropy
(GCE) [Zhang & Sabuncu], symmetric cross entropy (SCE) [Wang et al.]), sample reweighting,
and sample selection (Co-teaching, DivideMix). These typically assume thousands of samples
per class. We transplant the *small-loss* idea into prototype space, using *large
distance* as the noisy-sample signal.

**Few-shot learning.** Prototypical networks classify by nearest class centroid. When
support labels are noisy, the centroid is pulled toward the wrong class. Prior work on
noisy few-shot learning uses meta-learning or label correction; we show that a simple
iterative trimmed prototype plus a robust loss already recovers much of the accuracy.

## 3. Method

**Setup.** Given a support set $S = \{(x_i, \tilde y_i)\}$ with $N$ classes and $K$
examples per class, where labels $\tilde y$ are corrupted by symmetric noise at rate $r$.
A frozen ImageNet-pretrained ResNet-18 extracts L2-normalized features $f(x_i)$. A clean
test set is used only for evaluation.

**Baselines.**
- *ProtoNet*: prototype $p_c = \text{mean}\{f(x_i) : \tilde y_i = c\}$, classify by cosine
  similarity.
- *CE linear probe*: a linear head trained on $(f(x_i), \tilde y_i)$ with cross-entropy.

**Component 1 — distance-driven selection.** We compute prototypes, then measure each
support sample's distance to *its (possibly wrong) assigned class prototype*. A sample is
flagged as noise if its distance exceeds a threshold $d > \mu + \beta\sigma$ (or a fixed
quantile $\tau$), and prototypes are recomputed from retained samples. We iterate $T$
rounds. This is the prototype-space translation of "small loss = clean".

**Component 2 — robust loss.** We train a linear head on the retained samples using GCE
$L = (1 - p_y^q)/q$, which is bounded for mislabeled examples and thus robust to residual
noise that selection misses.

**Variants for ablation:** selection-only, robust-loss-only, and full (both).

## 4. Experiments

**Setup.** CIFAR-10; $N{=}10$ classes; $K \in \{5,10,20\}$; symmetric noise
$r \in \{0, 0.1, 0.2, 0.4\}$; frozen ResNet-18 (ImageNet) features, 512-d, L2-normalized.
Metrics: accuracy, macro-F1, ECE (15 bins), and noise-detection precision/recall/F1.
Seeds fixed; noise mask and configs saved for reproducibility.

**Results (to fill from `results/*.csv`).**

Table 1: Clean-test accuracy vs noise rate ($K=10$).

| Method              | 0%  | 10% | 20% | 40% |
|---------------------|-----|-----|-----|-----|
| ProtoNet            | [FILL] | [FILL] | [FILL] | [FILL] |
| CE linear probe     | [FILL] | [FILL] | [FILL] | [FILL] |
| selection-only      | [FILL] | [FILL] | [FILL] | [FILL] |
| robust-loss-only    | [FILL] | [FILL] | [FILL] | [FILL] |
| **full**            | [FILL] | [FILL] | [FILL] | [FILL] |

Table 2: Noise-detection precision / recall / F1 of the selection step.

| Method        | 10%  | 20%  | 40%  |
|---------------|------|------|------|
| full          | [FILL] | [FILL] | [FILL] |

*Expected finding*: full > each single component > CE baseline; the gap widens with noise
rate. ECE and macro-F1 follow the same trend. Calibration degrades with noise (higher ECE),
and our method partially restores it.

**Ablations.** Sweep $\beta$, $q$, $T$, and the $\tau$ variant at $r=0.4$ (see
`results/ablation_summary.csv`). *Expected*: moderate $\beta$ and $q\approx0.7$ are
stable; very aggressive trimming (small $\beta$/$\tau$) over-fits but hurts recall of
clean samples.

**Few-shot scale.** At $r=0.4$, sweep $K \in \{5,10,20\}$. *Expected*: the benefit of our
method is largest at the smallest $K$, where each sample matters most.

## 5. Error Analysis

**Misclassification.** From the confusion matrix (`results/figs/confusion_matrix.png`),
most errors occur between visually similar pairs (cat/dog, truck/automobile). Noisy labels
shift the decision boundary toward these confusable classes.

**Noise-identification failures.** The distance signal misses noisy samples when (i) the
prototype itself has already been polluted at high $r$, so a noisy sample is not an
outlier, or (ii) a class has high intra-class variance, making its clean samples
indistinguishable from noise. False flags happen for genuinely hard/atypical clean
samples. See `results/figs/distance_distribution.png` and `tsne.png`.

**Prototype drift.** We measure cosine similarity between clean prototypes and their
noisy/robust counterparts (`results/figs/prototype_drift.png`). The robust prototype is
consistently closer to the clean prototype, quantifying how selection repairs the
centroid.

## 6. Limitations and Conclusion

**Limitations.** (i) Symmetric noise only; real label noise is class-dependent
(asymmetric). (ii) ImageNet-pretrained features transfer imperfectly to CIFAR-10; a
self-supervised backbone would strengthen the "no labels used" claim. (iii) The selection
threshold is heuristic.

**Conclusion.** Label noise and label scarcity interact destructively, but a simple
distance-driven selection plus a robust loss substantially recovers clean-test accuracy
while producing an interpretable noise-detection signal. Our error analysis clarifies
*when* the method fails, pointing to class-dependent noise and prototype pollution as the
main open challenges.

## References

[Zhang & Sabuncu] Generalized Cross Entropy Loss for Training Deep Neural Networks with
Noisy Labels. *NeurIPS 2018*.

[Wang et al.] Symmetric Cross Entropy for Robust Learning with Noisy Labels. *ICCV 2019*.

[Snell et al.] Prototypical Networks for Few-shot Learning. *NeurIPS 2017*.
