"""Phase 3a: Bayesian Decision Theory for Comma Insertion

假說：正常段中模型輸出逗號的位置包含可學的統計信號。
用 logistic regression 從 logit probe 數據學 P(該斷句|特徵)，
再用非對稱損失函數決定最佳閾值，畫 Pareto front。

預期結果：
- 若 AUC > 0.8 → 特徵有區分力，可建有效的 LogitsProcessor
- 若 AUC < 0.7 → 特徵不足，需要更深層語言知識（進 Phase 3c）

特徵：
- chars_since_punct: 距上一個標點的字元數
- comma_prob: 逗號 token 的機率
- comma_rank: 逗號 token 的排名
- winner_prob: 實際被選 token 的機率
- winner_comma_gap: 被選 token 與逗號的 log-prob 差
- position_ratio: 目前步驟在段中的位置比例
- prev_is_punct: 上一步是否為標點
"""

import json
import sys
import math
import os
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

PROBE_PATH = Path("experiments/results/comma_logit_probe.json")
OUTPUT_PATH = Path("experiments/results/experiment_bayesian_comma.json")

COMMA_TID = 11
PUNCT_TOKENS = {11, 1543, 1231, 8, 30, 7, 0}  # , 。 、 ? ! . (approx)

def is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)

def is_punct(token_str):
    return token_str in {",", "，", "。", "？", "！", "、", ".", "?", "!"}


def extract_features(segments):
    """Extract features from logit probe data.
    Returns (X, y, meta) where meta contains segment/step info for tracing.
    """
    X = []
    y = []
    meta = []

    for seg in segments:
        steps = seg["steps"]
        chars_since_punct = 0
        cjk_since_punct = 0

        for i, step in enumerate(steps):
            actual_token = step["actual_token"]
            comma_prob = step["comma_hw"]["prob"]
            comma_rank = step["comma_hw"]["rank"]

            # Winner info (actual selected token)
            winner_prob = step["top5"][0]["prob"] if step["top5"] else 0.5
            winner_comma_gap = math.log(winner_prob + 1e-10) - math.log(comma_prob + 1e-10)

            # Position in segment
            position_ratio = i / max(len(steps) - 1, 1)

            # Previous token info
            prev_is_punct = 0
            if i > 0:
                prev_token = steps[i - 1]["actual_token"]
                prev_is_punct = 1 if is_punct(prev_token) else 0

            # Period probability (competing punctuation)
            period_prob = step["period"]["prob"]

            features = {
                "chars_since_punct": chars_since_punct,
                "cjk_since_punct": cjk_since_punct,
                "comma_prob": comma_prob,
                "comma_rank": comma_rank,
                "comma_log_prob": math.log(comma_prob + 1e-10),
                "winner_prob": winner_prob,
                "winner_comma_gap": winner_comma_gap,
                "position_ratio": position_ratio,
                "prev_is_punct": prev_is_punct,
                "period_prob": period_prob,
                "period_log_prob": math.log(period_prob + 1e-10),
            }

            # Label: 1 if model actually output comma here
            label = 1 if step["actual_tid"] == COMMA_TID else 0

            X.append(features)
            y.append(label)
            meta.append({
                "seg_wav": seg["wav"],
                "step": i,
                "actual_token": actual_token,
                "commas_in_seg": seg["commas"],
                "cjk_chars": seg["cjk_chars"],
                "is_problem_seg": seg["commas"] == 0 and seg["cjk_chars"] > 20,
            })

            # Update running counters
            if is_punct(actual_token):
                chars_since_punct = 0
                cjk_since_punct = 0
            else:
                chars_since_punct += 1
                if any(is_cjk(c) for c in actual_token):
                    cjk_since_punct += 1

    return X, y, meta


def logistic_regression_fit(X, y, lr=0.1, epochs=1000, l2=0.01):
    """Pure Python logistic regression (no sklearn dependency).
    Returns weights dict and bias.
    """
    feature_names = list(X[0].keys())
    n_features = len(feature_names)
    n_samples = len(X)

    # Normalize features
    means = {f: sum(x[f] for x in X) / n_samples for f in feature_names}
    stds = {}
    for f in feature_names:
        var = sum((x[f] - means[f]) ** 2 for x in X) / n_samples
        stds[f] = max(var ** 0.5, 1e-8)

    # Convert to normalized arrays
    X_norm = [[((x[f] - means[f]) / stds[f]) for f in feature_names] for x in X]

    # Initialize weights
    w = [0.0] * n_features
    b = 0.0

    def sigmoid(z):
        if z > 30: return 1.0
        if z < -30: return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    # Gradient descent
    for epoch in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        total_loss = 0.0

        for i in range(n_samples):
            z = sum(w[j] * X_norm[i][j] for j in range(n_features)) + b
            pred = sigmoid(z)
            error = pred - y[i]

            for j in range(n_features):
                grad_w[j] += error * X_norm[i][j] + l2 * w[j]
            grad_b += error

            # Log loss
            if y[i] == 1:
                total_loss -= math.log(pred + 1e-10)
            else:
                total_loss -= math.log(1 - pred + 1e-10)

        # Update
        for j in range(n_features):
            w[j] -= lr / n_samples * grad_w[j]
        b -= lr / n_samples * grad_b

        if epoch % 200 == 0:
            avg_loss = total_loss / n_samples
            print(f"  epoch {epoch}: loss={avg_loss:.4f}")

    # Convert back to original scale weights for interpretability
    weights_orig = {}
    for j, f in enumerate(feature_names):
        weights_orig[f] = w[j] / stds[f]
    bias_orig = b - sum(w[j] * means[f] / stds[f] for j, f in enumerate(feature_names))

    return weights_orig, bias_orig, means, stds, w, b


def predict_proba(x, weights, bias, means, stds, w_norm, b_norm):
    """Predict probability for a single sample."""
    feature_names = list(weights.keys())
    z = sum(w_norm[j] * ((x[f] - means[f]) / stds[f]) for j, f in enumerate(feature_names)) + b_norm
    if z > 30: return 1.0
    if z < -30: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_auc(y_true, y_scores):
    """Compute AUC-ROC (pure Python)."""
    pairs = sorted(zip(y_scores, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr

    return auc


def compute_metrics_at_threshold(y_true, y_scores, threshold, meta):
    """Compute problem improvement and normal side effects at a given threshold."""
    # Group by segment
    seg_results = defaultdict(lambda: {"steps": [], "is_problem": False})
    for i, (yt, ys, m) in enumerate(zip(y_true, y_scores, meta)):
        wav = m["seg_wav"]
        seg_results[wav]["steps"].append((yt, ys, m))
        seg_results[wav]["is_problem"] = m["is_problem_seg"]

    problem_improved = 0
    problem_total = 0
    normal_affected = 0
    normal_total = 0

    for wav, seg_data in seg_results.items():
        # Count how many steps the classifier would boost (predict comma)
        predicted_commas = sum(1 for yt, ys, m in seg_data["steps"] if ys >= threshold)
        actual_commas = sum(1 for yt, ys, m in seg_data["steps"] if yt == 1)

        if seg_data["is_problem"]:
            problem_total += 1
            if predicted_commas > 0:
                problem_improved += 1
        elif actual_commas > 0:  # normal segment (has commas)
            normal_total += 1
            # "affected" = classifier predicts comma where model didn't output one
            false_positives = sum(1 for yt, ys, m in seg_data["steps"]
                                if ys >= threshold and yt == 0)
            if false_positives > 0:
                normal_affected += 1

    return {
        "threshold": threshold,
        "problem_improved": problem_improved,
        "problem_total": problem_total,
        "problem_rate": problem_improved / max(problem_total, 1),
        "normal_affected": normal_affected,
        "normal_total": normal_total,
        "normal_rate": normal_affected / max(normal_total, 1),
    }


def main():
    print("=" * 60)
    print("Phase 3a: Bayesian Decision — Logistic Regression")
    print("=" * 60)

    # Load data
    with open(PROBE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["logit_probe"]
    print(f"載入 {len(segments)} 段")

    # Split into normal and problem segments
    normal_segs = [s for s in segments if not (s["commas"] == 0 and s["cjk_chars"] > 20)]
    problem_segs = [s for s in segments if s["commas"] == 0 and s["cjk_chars"] > 20]
    print(f"正常段: {len(normal_segs)}, 問題段: {len(problem_segs)}")

    # Extract features from ALL segments
    print("\n--- 特徵提取 ---")
    X, y, meta = extract_features(segments)
    print(f"總步驟: {len(X)}, 正樣本(逗號): {sum(y)}, 負樣本: {len(y)-sum(y)}")
    print(f"正樣本比例: {sum(y)/len(y)*100:.1f}%")

    # Feature importance preview
    print("\n--- 特徵分佈（正樣本 vs 負樣本）---")
    feature_names = list(X[0].keys())
    for f in feature_names:
        pos_vals = [x[f] for x, yi in zip(X, y) if yi == 1]
        neg_vals = [x[f] for x, yi in zip(X, y) if yi == 0]
        pos_mean = sum(pos_vals) / max(len(pos_vals), 1)
        neg_mean = sum(neg_vals) / max(len(neg_vals), 1)
        print(f"  {f:25s}  逗號={pos_mean:10.4f}  非逗號={neg_mean:10.4f}  差={pos_mean-neg_mean:+.4f}")

    # Train logistic regression
    print("\n--- 訓練 Logistic Regression ---")
    weights, bias, means, stds, w_norm, b_norm = logistic_regression_fit(X, y, lr=0.5, epochs=2000, l2=0.01)

    # Feature importance (by absolute weight)
    print("\n--- 特徵重要性（|weight|排序）---")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for f, w in sorted_weights:
        direction = "+" if w > 0 else "-"
        print(f"  {direction} {f:25s}  weight={w:+.4f}")

    # Predict probabilities
    print("\n--- 模型評估 ---")
    y_scores = [predict_proba(x, weights, bias, means, stds, w_norm, b_norm) for x in X]

    # AUC
    auc = compute_auc(y, y_scores)
    print(f"AUC-ROC: {auc:.4f}")

    if auc < 0.7:
        print("⚠ AUC < 0.7：特徵區分力不足")
    elif auc < 0.8:
        print("△ AUC 0.7-0.8：有一定區分力但不強")
    else:
        print("✓ AUC > 0.8：特徵有良好區分力")

    # Precision-Recall at various thresholds
    print("\n--- 不同閾值的精確度/召回率 ---")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds:
        tp = sum(1 for yi, si in zip(y, y_scores) if si >= t and yi == 1)
        fp = sum(1 for yi, si in zip(y, y_scores) if si >= t and yi == 0)
        fn = sum(1 for yi, si in zip(y, y_scores) if si < t and yi == 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        print(f"  t={t:.2f}  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}  (TP={tp} FP={fp})")

    # Pareto front: threshold → (problem improvement, normal side effects)
    print("\n--- Pareto Front ---")
    pareto_thresholds = [i * 0.02 for i in range(1, 50)]
    pareto_points = []
    for t in pareto_thresholds:
        metrics = compute_metrics_at_threshold(y, y_scores, t, meta)
        pareto_points.append(metrics)

    print(f"{'threshold':>10}  {'問題改善%':>10}  {'正常副作用%':>12}  {'改善段':>6}  {'影響段':>6}")
    print("-" * 55)
    for p in pareto_points:
        if p["problem_rate"] > 0 or p["threshold"] in [0.1, 0.3, 0.5]:
            print(f"  {p['threshold']:.2f}       {p['problem_rate']*100:5.1f}%       {p['normal_rate']*100:5.1f}%      {p['problem_improved']:3d}      {p['normal_affected']:3d}")

    # Find knee point (best tradeoff)
    print("\n--- 尋找最佳閾值（拐點）---")
    best_score = -1
    best_threshold = 0.5
    for p in pareto_points:
        # Score = improvement - 2 * side_effect (asymmetric: side effects cost more)
        score = p["problem_rate"] - 2.0 * p["normal_rate"]
        if score > best_score:
            best_score = score
            best_threshold = p["threshold"]

    best_metrics = compute_metrics_at_threshold(y, y_scores, best_threshold, meta)
    print(f"最佳閾值: {best_threshold:.2f}")
    print(f"問題段改善: {best_metrics['problem_rate']*100:.1f}% ({best_metrics['problem_improved']}/{best_metrics['problem_total']})")
    print(f"正常段副作用: {best_metrics['normal_rate']*100:.1f}% ({best_metrics['normal_affected']}/{best_metrics['normal_total']})")

    # Comparison with Phase 1 best (t15_a0.2_m3.0: 51.4% improvement, 21.7% side effects)
    print("\n--- 與 Phase 1 最佳比較 ---")
    print(f"Phase 1 (t15_a0.2): 改善 51.4%, 副作用 21.7%")
    print(f"Phase 3a (t={best_threshold:.2f}): 改善 {best_metrics['problem_rate']*100:.1f}%, 副作用 {best_metrics['normal_rate']*100:.1f}%")
    if best_metrics["problem_rate"] > 0.5 and best_metrics["normal_rate"] < 0.1:
        print("✓ 達成成功標準（改善>50%, 副作用<10%）")
    else:
        print("✗ 未達成功標準")

    # Sample predictions on problem segments
    print("\n--- 問題段預測抽樣 ---")
    problem_step_scores = [(m, ys) for m, ys in zip(meta, y_scores) if m["is_problem_seg"]]
    high_score_steps = sorted(problem_step_scores, key=lambda x: x[1], reverse=True)[:20]
    print("問題段中模型預測最應該是逗號的位置：")
    for m, score in high_score_steps[:10]:
        print(f"  score={score:.3f}  step={m['step']}  token='{m['actual_token']}'  seg={m['seg_wav'].split(os.sep)[-2]}")

    # Save results
    output = {
        "meta": {
            "method": "logistic_regression",
            "date": "2026-07-01",
            "total_steps": len(X),
            "positive_samples": sum(y),
            "negative_samples": len(y) - sum(y),
            "auc_roc": auc,
        },
        "feature_importance": sorted_weights,
        "weights": weights,
        "bias": bias,
        "normalization": {"means": means, "stds": stds},
        "pareto_front": pareto_points,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
