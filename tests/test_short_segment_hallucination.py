"""短音訊幻覺處理回歸測試。

資料來自 `tests/data/short_segment_cases.json`：從 `data/logs/history.sqlite3` 抽出
143 筆 < 3 秒的真實辨識嘗試，label 為人工判讀（產生器見
`experiments/extract_short_segment_cases.py`）。

背景：`min_hallucination_audio_sec = 3.0` 原本讓 3 秒以下的段跳過整個幻覺偵測，
`docs/roadmap.md` 記錄「3 秒以下一律不檢查必須重驗，先從現有短段 log 建立測試」。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from asr_input.audio.streaming_vad import StreamingVAD
from asr_input.streaming import StreamingSession

CASES_PATH = Path(__file__).parent / "data" / "short_segment_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
THRESHOLD_SEC = 1.5
MIN_RECOVERABLE_SEC = 3.0


class FakeVADModel:
    def reset_states(self):
        pass


class ScriptedEngine:
    """Replays recorded (text, transcribe_sec) pairs instead of running a model."""

    def __init__(self, results):
        self._results = iter(results)
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000):  # pragma: no cover - unused
        raise AssertionError("scripted engine is driven through _transcribe_one")


class PassThroughPipeline:
    def run(self, text):
        return text


def make_session(engine, **kwargs):
    return StreamingSession(
        engine=engine,
        pipeline=PassThroughPipeline(),
        vad_model=FakeVADModel(),
        sample_rate=16000,
        hallucination_threshold_sec=THRESHOLD_SEC,
        min_hallucination_audio_sec=MIN_RECOVERABLE_SEC,
        **kwargs,
    )


def run_one_segment(monkeypatch, audio_sec, transcribe_sec, text):
    """Feed one segment whose transcription latency and text are pre-recorded."""
    session = make_session(ScriptedEngine([]))
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: (text, transcribe_sec))
    monkeypatch.setattr(session, "_try_rms_normalize", lambda audio: None)
    samples = max(1, int(audio_sec * 16000))
    audio = np.full(samples, 0.1, dtype=np.float32)
    session.start_from_segments([(audio, [0.9])])
    return session.stop(), session.segment_stats[0]


def test_recorded_hallucinations_are_separable_by_latency():
    """凍結資料上，1.5s 延遲門檻對 <3s 段的分離能力（含已知漏抓）。"""
    flagged = [c for c in CASES if c["transcribe_sec"] > THRESHOLD_SEC]
    normal_flagged = [c for c in flagged if c["label"] == "normal"]
    missed = [
        c for c in CASES if c["label"] == "hallucination" and c["transcribe_sec"] <= THRESHOLD_SEC
    ]

    # 沒有任何一段被判定為可用（normal）的輸出被誤殺
    assert normal_flagged == []
    # 已知漏抓只有 'Duh.'（0.8s→1.03s），延遲落在正常範圍內
    assert [c["raw"] for c in missed] == ["Duh."]


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["label"] == "hallucination" and c["transcribe_sec"] > THRESHOLD_SEC],
    ids=lambda c: c["source"],
)
def test_recorded_short_hallucinations_are_dropped(monkeypatch, case):
    text, stat = run_one_segment(
        monkeypatch, case["audio_sec"], case["transcribe_sec"], case["raw"]
    )

    assert text == ""
    assert stat["rejected"] is True
    assert stat["raw"] == case["raw"]  # 證據保留在 log，只是不進輸出


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["label"] == "normal"][:20],
    ids=lambda c: c["source"],
)
def test_recorded_normal_short_segments_still_pass(monkeypatch, case):
    text, stat = run_one_segment(
        monkeypatch, case["audio_sec"], case["transcribe_sec"], case["raw"]
    )

    assert text == case["raw"]
    assert not stat.get("rejected")


def test_long_slow_segment_is_kept_not_dropped(monkeypatch):
    """長段仍有重切空間，即使偏慢也不可直接丟掉使用者的內容。"""
    session = make_session(ScriptedEngine([]))
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: ("一段很慢但真實的話", 2.0))
    monkeypatch.setattr(session, "_try_rms_normalize", lambda audio: None)
    monkeypatch.setattr(StreamingVAD, "resegment", staticmethod(lambda audio, probs, **kw: []))
    audio = np.full(16000 * 10, 0.1, dtype=np.float32)

    session.start_from_segments([(audio, [0.9])])

    assert session.stop() == "一段很慢但真實的話"
    assert not session.segment_stats[0].get("rejected")


def test_exhausted_fallback_drops_still_hallucinating_short_subsegments(monkeypatch):
    """重現 job 364bdc50：切分用盡後，仍幻覺的極短子段曾被原樣採用。"""
    session = make_session(ScriptedEngine([]), fallback_silence_ms=[500, 500])
    monkeypatch.setattr(
        StreamingVAD,
        "resegment",
        staticmethod(
            lambda audio, probs, **kw: [
                np.full(int(0.29 * 16000), 0.1, dtype=np.float32),
                np.full(int(2.6 * 16000), 0.1, dtype=np.float32),
            ]
        ),
    )
    scripted = iter(
        [
            ("原始長段輸出", 4.0),  # 觸發幻覺偵測
            ("作詞・作曲・編曲 男高等部分", 3.21),  # 0.29s 子段 → 應捨棄
            ("邏輯之類的 反正", 0.25),  # 2.6s 子段 → 應保留
        ]
    )
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: next(scripted))
    monkeypatch.setattr(session, "_try_rms_normalize", lambda audio: None)

    session.start_from_segments([(np.full(16000 * 5, 0.1, dtype=np.float32), [0.9])])

    assert session.stop() == "邏輯之類的 反正"
    stat = session.segment_stats[0]
    assert stat["fallback_exhausted"] is True
    assert stat["rejected_sub_segments"] == 1
    assert stat["processed"] == "邏輯之類的 反正"
    assert [
        attempt["adopted"] for attempt in stat["attempts"] if attempt["kind"] == "resegment"
    ] == [False, True]


def test_short_segment_still_gets_rms_normalize_before_being_dropped(monkeypatch):
    """短段先給 RMS 正規化一次機會，救回來就不該丟。"""
    session = make_session(ScriptedEngine([]))
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: ("幻覺輸出", 3.0))
    monkeypatch.setattr(
        session,
        "_try_rms_normalize",
        lambda audio: (audio, "救回來的短句", 0.4),
    )

    session.start_from_segments([(np.full(int(0.5 * 16000), 0.01, dtype=np.float32), [0.9])])

    assert session.stop() == "救回來的短句"
    assert not session.segment_stats[0].get("rejected")


def test_remote_engine_latency_never_drops_text(monkeypatch):
    """雲端引擎的 wall-clock 延遲不是品質訊號，不得據此丟掉文字。"""

    class RemoteEngine(ScriptedEngine):
        transcription_latency_is_quality_signal = False

    session = make_session(RemoteEngine([]))
    monkeypatch.setattr(session, "_transcribe_one", lambda audio: ("遠端結果", 9.0))

    session.start_from_segments([(np.full(int(0.5 * 16000), 0.1, dtype=np.float32), [0.9])])

    assert session.stop() == "遠端結果"
    assert not session.segment_stats[0].get("rejected")
