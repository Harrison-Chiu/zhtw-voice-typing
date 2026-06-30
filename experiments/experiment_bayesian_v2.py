"""Phase 3a-v2: Bayesian Decision with LINGUISTIC features only

v1 的問題：comma_prob 主導一切，分類器只複製模型自己的判斷。
v2 假說：逗號位置有可學的語言模式（字元距離、前文詞彙），獨立於模型的逗號機率。
若純語言特徵能預測逗號位置（AUC>0.7），代表有模型未充分利用的語言信號可用於修正。
若不能（AUC<0.7），代表逗號決策主要靠聲學，文字端無法補救。

新增特徵：
- 前文 token 是否為常見的逗號前詞（「的」「了」「嗎」「吧」等）
- 前文 2-gram 模式
- 段內已出現的逗號數
"""

import json
import sys
import math
import os
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

PROBE_PATH = Path("experiments/results/comma_logit_probe.json")
OUTPUT_PATH = Path("experiments/results/experiment_bayesian_v2.json")
COMMA_TID = 11


def is_cjk(ch):
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def is_punct(token_str):
    return token_str in {",", "，", "。", "？", "！", "、", ".", "?", "!"}


# Common tokens that precede commas in Chinese (clause-ending particles, connectives)
CLAUSE_END_CHARS = {"的", "了", "嗎", "吧", "啊", "呢", "喔", "耶", "啦", "嘛", "哦"}
CONNECTIVE_CHARS = {"但", "而", "且", "或", "就", "也", "還", "所", "因"}


def extract_linguistic_features(segments):
    """Extract only linguistic features (no model probability features)."""
    X = []
    y = []
    meta = []

    # First pass: collect bigram statistics
    bigram_before_comma = Counter()
    bigram_total = Counter()
    for seg in segments:
        if seg["commas"] == 0 and seg["cjk_chars"] > 20:
            continue  # Only learn from normal segments
        steps = seg["steps"]
        for i in range(1, len(steps)):
            prev_tok = steps[i - 1]["actual_token"]
            bigram_total[prev_tok] += 1
            if steps[i]["actual_tid"] == COMMA_TID:
                bigram_before_comma[prev_tok] += 1

    # Compute P(comma | prev_token) for common tokens
    token_comma_prob = {}
    for tok, count in bigram_total.items():
        if count >= 5:
            token_comma_prob[tok] = bigram_before_comma.get(tok, 0) / count

    print(f"建立了 {len(token_comma_prob)} 個 token 的逗號先驗")
    top_comma_tokens = sorted(token_comma_prob.items(), key=lambda x: x[1], reverse=True)[:20]
    print("最常接逗號的 token：")
    for tok, prob in top_comma_tokens:
        print(f"  '{tok}' → P(comma)={prob:.3f} (n={bigram_total[tok]})")

    # Second pass: extract features
    for seg in segments:
        steps = seg["steps"]
        chars_since_punct = 0
        cjk_since_punct = 0
        commas_so_far = 0

        for i, step in enumerate(steps):
            actual_token = step["actual_token"]

            # === LINGUISTIC FEATURES ONLY ===
            # 1. Distance features
            f_chars_since = chars_since_punct
            f_cjk_since = cjk_since_punct

            # 2. Position
            f_position = i / max(len(steps) - 1, 1)

            # 3. Previous token patterns
            prev_token = steps[i - 1]["actual_token"] if i > 0 else ""
            f_prev_is_clause_end = 1 if any(c in CLAUSE_END_CHARS for c in prev_token) else 0
            f_prev_is_connective = 1 if any(c in CONNECTIVE_CHARS for c in prev_token) else 0
            f_prev_is_punct = 1 if is_punct(prev_token) else 0
            f_prev_comma_prior = token_comma_prob.get(prev_token, 0.0)

            # 4. Two-step lookback
            prev2_token = steps[i - 2]["actual_token"] if i >= 2 else ""
            f_prev2_comma_prior = token_comma_prob.get(prev2_token, 0.0)

            # 5. Segment-level context
            f_commas_so_far = commas_so_far
            f_seg_length = len(steps)
            f_comma_density_so_far = commas_so_far / max(i, 1)

            # 6. Derived: is it "overdue" for a comma?
            f_overdue = max(0, cjk_since_punct - 8)  # average comma interval ~8 chars

            # === Minimal model features (only comma_prob as a reference) ===
            # We include comma_prob but with a twist: we also compute the
            # "surprise" — how much higher comma_prob is relative to baseline
            f_comma_prob = step["comma_hw"]["prob"]
            f_comma_rank = step["comma_hw"]["rank"]

            features = {
                "chars_since_punct": f_chars_since,
                "cjk_since_punct": f_cjk_since,
                "overdue": f_overdue,
                "position_ratio": f_position,
                "prev_clause_end": f_prev_is_clause_end,
                "prev_connective": f_prev_is_connective,
                "prev_is_punct": f_prev_is_punct,
                "prev_comma_prior": f_prev_comma_prior,
                "prev2_comma_prior": f_prev2_comma_prior,
                "commas_so_far": f_commas_so_far,
                "seg_length": f_seg_length,
                "comma_density_so_far": f_comma_density_so_far,
            }

            label = 1 if step["actual_tid"] == COMMA_TID else 0

            X.append(features)
            y.append(label)
            meta.append({
                "seg_wav": seg["wav"],
                "step": i,
                "actual_token": actual_token,
                "prev_token": prev_token,
                "comma_prob": f_comma_prob,
                "comma_rank": f_comma_rank,
                "is_problem_seg": seg["commas"] == 0 and seg["cjk_chars"] > 20,
            })

            if is_punct(actual_token):
                chars_since_punct = 0
                cjk_since_punct = 0
                commas_so_far += (1 if actual_token in {",", "，"} else 0)
            else:
                chars_since_punct += 1
                if any(is_cjk(c) for c in actual_token):
                    cjk_since_punct += 1

    return X, y, meta, token_comma_prob


def logistic_regression_fit(X, y, lr=0.3, epochs=3000, l2=0.01):
    """Pure Python logistic regression."""
    feature_names = list(X[0].keys())
    n_features = len(feature_names)
    n_samples = len(X)

    means = {f: sum(x[f] for x in X) / n_samples for f in feature_names}
    stds = {}
    for f in feature_names:
        var = sum((x[f] - means[f]) ** 2 for x in X) / n_samples
        stds[f] = max(var ** 0.5, 1e-8)

    X_norm = [[(x[f] - means[f]) / stds[f] for f in feature_names] for x in X]
    w = [0.0] * n_features
    b = 0.0

    def sigmoid(z):
        if z > 30: return 1.0
        if z < -30: return 0.0
        return 1.0 / (1.0 + math.exp(-z))

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
            total_loss -= (y[i] * math.log(pred + 1e-10) + (1 - y[i]) * math.log(1 - pred + 1e-10))

        for j in range(n_features):
            w[j] -= lr / n_samples * grad_w[j]
        b -= lr / n_samples * grad_b

        if epoch % 500 == 0:
            print(f"  epoch {epoch}: loss={total_loss/n_samples:.4f}")

    weights_orig = {}
    for j, f in enumerate(feature_names):
        weights_orig[f] = w[j] / stds[f]
    bias_orig = b - sum(w[j] * means[f] / stds[f] for j, f in enumerate(feature_names))

    return weights_orig, bias_orig, means, stds, w, b


def predict_proba(x, weights, bias, means, stds, w_norm, b_norm):
    feature_names = list(weights.keys())
    z = sum(w_norm[j] * ((x[f] - means[f]) / stds[f]) for j, f in enumerate(feature_names)) + b_norm
    if z > 30: return 1.0
    if z < -30: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_auc(y_true, y_scores):
    pairs = sorted(zip(y_scores, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fpr = prev_tpr = 0.0
    for score, label in pairs:
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


def main():
    print("=" * 60)
    print("Phase 3a-v2: Linguistic Features Only")
    print("=" * 60)

    with open(PROBE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["logit_probe"]
    print(f"載入 {len(segments)} 段")

    # Extract features
    print("\n--- 特徵提取 ---")
    X, y, meta, token_comma_prob = extract_linguistic_features(segments)
    print(f"總步驟: {len(X)}, 正樣本: {sum(y)}, 負樣本: {len(y)-sum(y)}")

    # Feature distribution
    print("\n--- 特徵分佈（正 vs 負）---")
    feature_names = list(X[0].keys())
    for f in feature_names:
        pos_vals = [x[f] for x, yi in zip(X, y) if yi == 1]
        neg_vals = [x[f] for x, yi in zip(X, y) if yi == 0]
        pos_mean = sum(pos_vals) / max(len(pos_vals), 1)
        neg_mean = sum(neg_vals) / max(len(neg_vals), 1)
        ratio = pos_mean / neg_mean if neg_mean != 0 else float('inf')
        print(f"  {f:25s}  逗號={pos_mean:8.3f}  非逗號={neg_mean:8.3f}  ratio={ratio:.2f}")

    # Train
    print("\n--- 訓練 ---")
    weights, bias, means, stds, w_norm, b_norm = logistic_regression_fit(X, y)

    print("\n--- 特徵重要性 ---")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for f, w in sorted_weights:
        d = "+" if w > 0 else "-"
        print(f"  {d} {f:25s}  weight={w:+.6f}")

    # Evaluate
    y_scores = [predict_proba(x, weights, bias, means, stds, w_norm, b_norm) for x in X]
    auc = compute_auc(y, y_scores)
    print(f"\nAUC-ROC (linguistic only): {auc:.4f}")

    # Compare: what if we ADD comma_prob back?
    print("\n--- 對比：加入 comma_prob ---")
    X_with_model = [{**x, "comma_prob": m["comma_prob"]} for x, m in zip(X, meta)]
    w2, b2, m2, s2, wn2, bn2 = logistic_regression_fit(X_with_model, y)
    y_scores2 = [predict_proba(x, w2, b2, m2, s2, wn2, bn2) for x in X_with_model]
    auc2 = compute_auc(y, y_scores2)
    print(f"AUC-ROC (linguistic + comma_prob): {auc2:.4f}")
    print(f"AUC 提升: {auc2 - auc:+.4f}")

    # Key question: in PROBLEM segments, what do the linguistic features predict?
    print("\n--- 問題段分析 ---")
    problem_scores = [(m, s) for m, s in zip(meta, y_scores) if m["is_problem_seg"]]
    normal_comma_scores = [(m, s) for m, s, yi in zip(meta, y_scores, y) if yi == 1]

    if problem_scores:
        problem_vals = [s for _, s in problem_scores]
        normal_comma_vals = [s for _, s in normal_comma_scores]
        print(f"問題段步驟預測分數: mean={sum(problem_vals)/len(problem_vals):.4f}, max={max(problem_vals):.4f}")
        print(f"正常逗號步驟預測分數: mean={sum(normal_comma_vals)/len(normal_comma_vals):.4f}")

        # Distribution of problem segment scores
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        print("\n問題段分數分佈：")
        for i in range(len(bins) - 1):
            count = sum(1 for s in problem_vals if bins[i] <= s < bins[i + 1])
            pct = count / len(problem_vals) * 100
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:4d} ({pct:.1f}%)")

    # Top predictions in problem segments
    print("\n問題段中語言特徵預測最高的位置：")
    top_problem = sorted(problem_scores, key=lambda x: x[1], reverse=True)[:15]
    for m, score in top_problem:
        print(f"  score={score:.4f}  prev='{m['prev_token']}'  cur='{m['actual_token']}'  "
              f"comma_prob={m['comma_prob']:.4f}  rank={m['comma_rank']}  "
              f"seg={os.path.basename(os.path.dirname(m['seg_wav']))}")

    # Save
    output = {
        "meta": {
            "method": "logistic_regression_linguistic_only",
            "date": "2026-07-01",
            "auc_linguistic": auc,
            "auc_with_model": auc2,
        },
        "feature_importance": sorted_weights,
        "token_comma_priors": dict(sorted(token_comma_prob.items(), key=lambda x: x[1], reverse=True)[:50]),
        "problem_segment_analysis": {
            "top_predictions": [
                {"score": score, "prev_token": m["prev_token"], "cur_token": m["actual_token"],
                 "comma_prob": m["comma_prob"], "comma_rank": m["comma_rank"]}
                for m, score in top_problem
            ]
        },
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果已存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
