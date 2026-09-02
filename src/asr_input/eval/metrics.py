"""Quality metrics for ASR benchmark runs (`docs/asr-benchmark-proposal.md` §3.2–3.3).

Pure functions over strings — no model, no audio, no I/O — so they can be unit
tested against hand-built pairs with known edit distances. That matters more than
it sounds: if the metric is wrong, every conclusion drawn from it is wrong, and a
wrong metric is much harder to notice than a wrong transcript.

Everything is built on one alignment (`align()`), so CER, the error-type
breakdown, punctuation F1 and the insertion runs used for hallucination counting
all describe the *same* alignment rather than three inconsistent ones.

Not included on purpose: any semantic-similarity or LLM-judged score. §3.3 rules
those out as a main score because they overlook numbers, negation and proper
nouns — the errors that actually cost the user an edit.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

from asr_input.eval.signals import dominant_char_ratio

Op = Literal["equal", "sub", "del", "ins"]

# Punctuation reported separately by §3.3. Half-width forms are folded onto their
# full-width counterpart first, so a run is not penalised twice for a punctuation
# style the normaliser is meant to fix.
PUNCT_FOLD = {",": "，", ".": "。", "?": "？", "!": "！", ":": "：", ";": "；"}
REPORTED_PUNCT = ("，", "。", "？", "、")

CJK = r"一-鿿㐀-䶿"
_WORD_RE = re.compile(rf"[{CJK}]|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|\S")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AlignOp:
    op: Op
    ref: str | None
    hyp: str | None
    ref_index: int | None
    hyp_index: int | None


@dataclass(frozen=True)
class ErrorRate:
    substitutions: int
    deletions: int
    insertions: int
    ref_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference unit. An empty reference with any output is 1.0."""
        if self.ref_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_length

    def as_dict(self) -> dict:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "ref_length": self.ref_length,
            "errors": self.errors,
            "rate": self.rate,
        }


def align(ref: list[str] | str, hyp: list[str] | str) -> list[AlignOp]:
    """Levenshtein alignment with unit costs, returned as an explicit op list.

    Ties are resolved substitution → deletion → insertion. The choice is
    arbitrary but fixed, so counts are reproducible across runs.
    """
    r = list(ref)
    h = list(hyp)
    n, m = len(r), len(h)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            same = r[i - 1] == h[j - 1]
            ops.append(AlignOp("equal" if same else "sub", r[i - 1], h[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(AlignOp("del", r[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, h[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def error_rate(ref: list[str] | str, hyp: list[str] | str) -> ErrorRate:
    ops = align(ref, hyp)
    counts = Counter(o.op for o in ops)
    return ErrorRate(
        substitutions=counts["sub"],
        deletions=counts["del"],
        insertions=counts["ins"],
        ref_length=len(list(ref)),
    )


# --- text normalisation ----------------------------------------------------


def normalize_text(text: str, to_traditional: Callable[[str], str] | None = None) -> str:
    """Fold away the differences Normalized CER is meant to ignore.

    Whitespace is removed entirely rather than collapsed: Whisper sometimes emits
    a space where a comma belongs, and keeping it would charge the same mistake
    twice (once as an insertion, once as the missing comma).
    """
    text = unicodedata.normalize("NFKC", text)
    text = _SPACE_RE.sub("", text)
    text = "".join(PUNCT_FOLD.get(ch, ch) for ch in text)
    if to_traditional is not None:
        text = to_traditional(text)
    return text


def strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def tokenize_mixed(text: str) -> list[str]:
    """CJK by character, latin words and numbers as single tokens (§3.2 MER).

    Without this a misheard English word costs as many errors as it has letters,
    which makes any mixed-language corpus look dominated by its English.
    """
    return _WORD_RE.findall(_SPACE_RE.sub(" ", text).strip())


# --- headline metrics ------------------------------------------------------


def strict_cer(ref: str, hyp: str) -> ErrorRate:
    """CER on raw text — punctuation, spacing and script differences all count."""
    return error_rate(ref, hyp)


def normalized_cer(
    ref: str, hyp: str, *, to_traditional: Callable[[str], str] | None = None
) -> ErrorRate:
    """CER after folding whitespace, punctuation style and (optionally) script."""
    return error_rate(
        strip_punctuation(normalize_text(ref, to_traditional)),
        strip_punctuation(normalize_text(hyp, to_traditional)),
    )


def mer(ref: str, hyp: str) -> ErrorRate:
    """Mixed error rate: CJK per character, latin/number per word."""
    return error_rate(tokenize_mixed(ref), tokenize_mixed(hyp))


# --- diagnostics -----------------------------------------------------------


@dataclass(frozen=True)
class PunctScore:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        got = self.true_positives + self.false_positives
        return self.true_positives / got if got else 1.0

    @property
    def recall(self) -> float:
        want = self.true_positives + self.false_negatives
        return self.true_positives / want if want else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def punctuation_f1(ref: str, hyp: str, marks: tuple[str, ...] = REPORTED_PUNCT) -> dict:
    """Per-mark precision/recall/F1, scored on the alignment, not on raw counts.

    Position matters: a comma the model put in the wrong clause is not a hit, and
    a bare count comparison would score it as one.
    """
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)
    ops = align(ref_n, hyp_n)

    scores: dict[str, PunctScore] = {}
    for mark in marks:
        tp = sum(1 for o in ops if o.op == "equal" and o.ref == mark)
        fn = sum(1 for o in ops if o.op in ("del", "sub") and o.ref == mark)
        fp = sum(1 for o in ops if o.op in ("ins", "sub") and o.hyp == mark)
        scores[mark] = PunctScore(tp, fp, fn)

    micro = PunctScore(
        sum(s.true_positives for s in scores.values()),
        sum(s.false_positives for s in scores.values()),
        sum(s.false_negatives for s in scores.values()),
    )
    return {"per_mark": {k: v.as_dict() for k, v in scores.items()}, "micro": micro.as_dict()}


def insertion_runs(ref: str, hyp: str, min_chars: int = 8) -> list[dict]:
    """Contiguous stretches present in the hypothesis and absent from the gold.

    This is the measurable half of "hallucination rate" (§3.3): long inserted runs
    are what a fabricated sentence looks like in an alignment. Short runs are
    ordinary insertion errors, hence the `min_chars` floor.
    """
    ops = align(normalize_text(ref), normalize_text(hyp))
    runs: list[dict] = []
    current: list[str] = []
    start: int | None = None
    for op in ops:
        if op.op == "ins":
            if start is None:
                start = op.hyp_index
            current.append(op.hyp or "")
        else:
            if current and len(current) >= min_chars:
                runs.append({"start": start, "length": len(current), "text": "".join(current)})
            current, start = [], None
    if current and len(current) >= min_chars:
        runs.append({"start": start, "length": len(current), "text": "".join(current)})
    return runs


def traditional_consistency(text: str, to_traditional: Callable[[str], str]) -> dict:
    """How much of the output is still simplified, per `to_traditional`.

    `converted_ratio` is what fraction of characters the converter would rewrite;
    it is a property of the converter as much as of the text, so report the
    converter alongside it.
    """
    converted = to_traditional(text)
    if len(converted) != len(text):
        # A converter that changes length (phrase-level substitution) cannot be
        # compared character-by-character; report only whether it changed.
        return {
            "chars": len(text),
            "converted_chars": None,
            "converted_ratio": None,
            "changed": converted != text,
            "length_changed": True,
        }
    changed = sum(1 for a, b in zip(text, converted, strict=True) if a != b)
    return {
        "chars": len(text),
        "converted_chars": changed,
        "converted_ratio": changed / len(text) if text else 0.0,
        "changed": changed > 0,
        "length_changed": False,
    }


def repetition_stats(text: str) -> dict:
    """Degeneration signals: longest identical-character run and top-char share."""
    longest, run, prev = 0, 0, None
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        longest = max(longest, run)
    char, ratio = dominant_char_ratio(text)
    return {
        "chars": len(text),
        "longest_char_run": longest,
        "dominant_char": char,
        "dominant_char_ratio": ratio,
    }


@dataclass
class DeterminismResult:
    runs: int
    unique_outputs: int
    exact_match_ratio: float
    modal_output: str | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "runs": self.runs,
            "unique_outputs": self.unique_outputs,
            "exact_match_ratio": self.exact_match_ratio,
            "modal_output": self.modal_output,
            "counts": self.counts,
        }


def determinism(outputs: list[str]) -> DeterminismResult:
    """Agreement across N repeats of the same input under the same settings.

    `exact_match_ratio` is the share of runs equal to the most common output, so
    1.0 means fully deterministic and 1/N means every run differed.
    """
    if not outputs:
        return DeterminismResult(runs=0, unique_outputs=0, exact_match_ratio=0.0)
    counts = Counter(outputs)
    modal, modal_count = counts.most_common(1)[0]
    return DeterminismResult(
        runs=len(outputs),
        unique_outputs=len(counts),
        exact_match_ratio=modal_count / len(outputs),
        modal_output=modal,
        counts=dict(counts),
    )


def empty_rate(outputs: list[str]) -> dict:
    empty = sum(1 for o in outputs if not o.strip())
    return {"total": len(outputs), "empty": empty, "rate": empty / len(outputs) if outputs else 0.0}


def evaluate_pair(
    ref: str, hyp: str, *, to_traditional: Callable[[str], str] | None = None
) -> dict:
    """All per-utterance metrics for one gold/hypothesis pair."""
    result = {
        "strict_cer": strict_cer(ref, hyp).as_dict(),
        "normalized_cer": normalized_cer(ref, hyp, to_traditional=to_traditional).as_dict(),
        "mer": mer(ref, hyp).as_dict(),
        "punctuation": punctuation_f1(ref, hyp),
        "insertion_runs": insertion_runs(ref, hyp),
        "repetition": repetition_stats(hyp),
    }
    if to_traditional is not None:
        result["traditional"] = traditional_consistency(hyp, to_traditional)
    return result
