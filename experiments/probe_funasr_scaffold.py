"""實驗：Fun-ASR-Nano 內部 prompt scaffolding 的「字體」是否影響繁體引導?（2026-06-29）

使用者假設：get_prompt(model.py:550) 的固定鷹架是簡體（「语音转写成{language}」「热词列表」
「请结合上下文信息」）。若把鷹架整段換繁體，language="繁體中文" 的引導會不會從「內容相依
（上一實驗僅 1/2 檔）」變穩定？

做法：runtime monkeypatch 內層 FunASRNano.get_prompt，A/B 同段比較：
  A 簡體鷹架（原版）+ language=繁體中文
  B 繁體鷹架（patch）+ language=繁體中文
  外加 baseline language=中文 當參考。
樣本：01/03/05 三段。只存原始輸出，簡體率回主環境算。結論邊界：此模型/這幾段。

用法：.venv-funasr\Scripts\python.exe experiments/probe_funasr_scaffold.py <Fun-ASR_repo>
"""

import io, json, subprocess, sys, tempfile, types
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
SEG = ROOT / "data/test_audio/segments"
OUT = ROOT / "experiments/results/probe_funasr_scaffold_raw_2026-06-29.json"
MODEL = "FunAudioLLM/Fun-ASR-Nano-2512"
FILES = ["01_純中文長句_自我介紹.wav", "03_中英混合_Arduino.wav", "05_數字混合_電池規格.wav"]


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


def trad_get_prompt(self, hotwords, language=None, itn=True):
    """原 get_prompt 的繁體鷹架版本（字面照搬，只換簡→繁）。"""
    if len(hotwords) > 0:
        hw = ", ".join(hotwords)
        prompt = "請結合上下文資訊，更加準確地完成語音轉寫任務。如果沒有相關資訊，我們會留空。\n\n\n**上下文資訊：**\n\n\n"
        prompt += f"熱詞列表：[{hw}]\n"
    else:
        prompt = ""
    prompt += "語音轉寫" if language is None else f"語音轉寫成{language}"
    if not itn:
        prompt += "，不進行文本規整"
    return prompt + "："


def main():
    repo = Path(sys.argv[1]).resolve()
    mp = repo / "model.py"
    sys.path.insert(0, str(repo))
    import torch
    from funasr import AutoModel

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    wavs = {f: to_wav(SEG / f) for f in FILES}
    model = AutoModel(
        model=MODEL,
        hub="hf",
        trust_remote_code=True,
        remote_code=str(mp),
        device=dev,
        disable_update=True,
    )
    inner = model.model
    orig = inner.get_prompt
    # 印出原鷹架長相佐證
    print("原鷹架 (language=繁體中文):", repr(orig(hotwords=[], language="繁體中文")))
    print(
        "繁鷹架 (language=繁體中文):",
        repr(trad_get_prompt(inner, hotwords=[], language="繁體中文")),
    )

    def run(label, lang, scaffold):
        if scaffold == "trad":
            inner.get_prompt = types.MethodType(trad_get_prompt, inner)
        else:
            inner.get_prompt = orig
        out = {}
        for f, w in wavs.items():
            r = model.generate(input=[w], cache={}, batch_size=1, language=lang, itn=True)
            out[f] = r[0].get("text", "")
        return out

    res = {"files": FILES, "runs": {}}
    res["runs"]["A_simp_scaffold_zhongwen"] = run("baseline 中文/簡鷹架", "中文", "simp")
    res["runs"]["B_simp_scaffold_fanti"] = run("繁體中文/簡鷹架", "繁體中文", "simp")
    res["runs"]["C_trad_scaffold_fanti"] = run("繁體中文/繁鷹架", "繁體中文", "trad")
    for name, out in res["runs"].items():
        print(f"\n=== {name} ===")
        for f in FILES:
            print(f"  [{f}] {out[f]}")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n存原始輸出 {OUT.name}")


main()
