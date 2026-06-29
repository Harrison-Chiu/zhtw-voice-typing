"""實驗：Fun-ASR-Nano 的 language(自由文字) 能否引導輸出繁體?（2026-06-29）

Fun-ASR-Nano 的 get_prompt 把 language 內插成「语音转写成{language}」(model.py:560)，
是自由文字，比 Qwen 受限清單更直接。測多個 language 值能否引導繁體；外加 hotword 組合。
跑在隔離環境 .venv-funasr + remote_code 指向官方 model.py。只存原始輸出，簡體率回主環境算。
交叉驗證：每 probe 跑兩短檔。結論邊界：僅此模型/這兩短檔/這些 probe。

用法：.venv-funasr\Scripts\python.exe experiments/steer_funasr_traditional.py <Fun-ASR_repo>
"""

import io, json, subprocess, sys, tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "data/test_audio"
OUT = ROOT / "experiments/results/steer_funasr_raw_2026-06-29.json"
MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
FILES = ["中英錄音測試.m4a", "機器人展示，語音轉錄測試.m4a"]


def to_wav(src, sr=16000):
    t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    t.close()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sr),
            t.name,
        ],
        check=True,
    )
    return t.name


# language 值 probes（自由文字）+ 一組 hotword 組合
LANG_PROBES = {
    "baseline_zhongwen": ("中文", None),
    "fanti_zh": ("繁體中文", None),
    "fanti_cn": ("繁体中文", None),
    "tw_fanti": ("台灣繁體中文", None),
    "traditional_en": ("Traditional Chinese", None),
    "zh_paren_fanti": ("中文（繁體）", None),
    "fanti_plus_hotword": ("繁體中文", ["台灣", "繁體", "影片"]),
}


def main():
    repo = Path(sys.argv[1]).resolve()
    mp = repo / "model.py"
    sys.path.insert(0, str(repo))
    import torch
    from funasr import AutoModel

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    wavs = {f: to_wav(AUD / f) for f in FILES}
    model = AutoModel(
        model=MODEL,
        hub="hf",
        trust_remote_code=True,
        remote_code=str(mp),
        device=dev,
        disable_update=True,
    )
    res = {"files": FILES, "probes": {}}
    for name, (lang, hot) in LANG_PROBES.items():
        pf = {}
        for f, w in wavs.items():
            kw = {"language": lang, "itn": True}
            if hot:
                kw["hotwords"] = hot
            try:
                r = model.generate(input=[w], cache={}, batch_size=1, **kw)
                text = r[0].get("text", "")
            except Exception as e:
                text = f"<ERROR {type(e).__name__}: {e}>"
            pf[f] = text
        res["probes"][name] = {"language": lang, "hotwords": hot, "per_file": pf}
        print(f"\n=== {name}  language={lang!r} hot={hot} ===")
        for f in FILES:
            print(f"  [{f}] {pf[f]}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存原始輸出 {OUT.name}")


main()
