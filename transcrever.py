"""Transcrição local confiável (pt-BR, Apple Silicon) — 3 motores + verificação cruzada.

Padrão: --motor whisperx (transcreve o texto num passe e alinha palavra por
palavra com Wav2vec2). Foi o único dos quatro caminhos testados em 10/09/2026
que devolveu o áudio inteiro sem perder fala. Ver memory/diretrizes-transcricao.md
pra medição completa e pra quando usar cada motor.

    python3 transcrever.py video.mp4 -o saida.json                # whisperx (padrão)
    python3 transcrever.py video.mp4 -o saida.json --motor janela # whisper.cpp deslizante
    python3 transcrever.py video.mp4 -o saida.json --motor vad    # whisper.cpp + VAD
    python3 transcrever.py video.mp4 -o saida.json --verificar    # roda 2 motores e diffa

Saída: {"words": [{"text", "start", "end"}, ...]} — mesmo formato que o resto do
pipeline (pack_transcripts.py, render.py, build de legenda) já espera.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
VENV_PY = AQUI / "venv" / "bin" / "python"

BIN = os.environ.get("WHISPER_CPP_BIN", "/Volumes/1TB Mac/whisper.cpp/build/bin/whisper-cli")
MODEL = os.environ.get("WHISPER_CPP_MODEL", "/Volumes/1TB Mac/whisper.cpp/models/ggml-large-v3.bin")
VAD_MODEL = os.environ.get("WHISPER_VAD_MODEL", "/Volumes/1TB Mac/whisper.cpp/models/ggml-silero-v5.1.2.bin")
DTW = {"large-v3-turbo": "large.v3.turbo", "large-v3": "large.v3"}

WIN, HOP = 24.0, 18.0   # janela deslizante: 24s de janela, 6s de sobreposição


# ---------------------------------------------------------------- utilidades

def to_wav(src: Path, dest: Path, start: float | None = None, dur: float | None = None) -> None:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)]
    subprocess.run(cmd, check=True)


def duration(src: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    return float(out.stdout.strip())


def dtw_preset() -> str | None:
    name = Path(MODEL).stem.lower()
    return next((p for n, p in DTW.items() if n in name), None)


def words_from_tokens(transcription: list[dict]) -> list[dict]:
    """Token stream do whisper.cpp -> palavras (token com espaço à esquerda abre palavra)."""
    points = []
    for seg in transcription:
        for tok in seg.get("tokens", []):
            text = tok.get("text", "")
            if not text or text.startswith("[_"):
                continue
            t_dtw = tok.get("t_dtw", -1)
            t = t_dtw / 100.0 if (t_dtw is not None and t_dtw >= 0) \
                else tok.get("offsets", {}).get("from", 0) / 1000.0
            points.append((text, t))
    words = []
    for text, t in points:
        if text.startswith(" ") or not words:
            words.append({"text": text.strip(), "start": t, "end": t})
        else:
            words[-1]["text"] += text
    for i in range(len(words) - 1):
        words[i]["end"] = min(words[i + 1]["start"], words[i]["start"] + 2.0)
    if words:
        words[-1]["end"] = words[-1]["start"] + 0.4
    return [w for w in words if w["text"]]


def whisper_cpp(wav: Path, language: str, vad: bool) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "out"
        cmd = [BIN, "-m", MODEL, "-f", str(wav), "-l", language,
               "-ojf", "-of", str(stem), "-np", "-nt"]
        if (p := dtw_preset()):
            cmd += ["-dtw", p]
        if vad and Path(VAD_MODEL).exists():
            # -vp 500: o padrão (30ms) come a primeira/última palavra de cada
            # segmento de fala. Medido no vídeo "3 skills", não chutado.
            cmd += ["--vad", "-vm", VAD_MODEL, "-vt", "0.35", "-vp", "500",
                    "-vsd", "300", "-vo", "0.4"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli falhou: {r.stderr[-800:]}")
        raw = json.loads(Path(str(stem) + ".json").read_text())
    return words_from_tokens(raw.get("transcription", []))


# ------------------------------------------------------------------- motores

def motor_whisperx(src: Path, language: str) -> list[dict]:
    """Melhor resultado medido. Precisa do venv (ver README)."""
    if not VENV_PY.exists():
        raise SystemExit(f"venv não encontrado em {VENV_PY} — ver ferramentas/transcrever/README.md")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "full.wav"
        to_wav(src, wav)
        out = Path(tmp) / "wx.json"
        script = f'''
import json, whisperx
audio = whisperx.load_audio({str(wav)!r})
m = whisperx.load_model("large-v3", "cpu", compute_type="int8", language={language!r})
r = m.transcribe(audio, batch_size=8)
am, meta = whisperx.load_align_model(language_code={language!r}, device="cpu")
r = whisperx.align(r["segments"], am, meta, audio, "cpu", return_char_alignments=False)
ws = [{{"text": w["word"], "start": round(w.get("start", 0), 2), "end": round(w.get("end", 0), 2)}}
      for s in r["segments"] for w in s.get("words", [])]
json.dump(ws, open({str(out)!r}, "w"), ensure_ascii=False)
'''
        r = subprocess.run([str(VENV_PY), "-c", script], capture_output=True, text=True)
        if not out.exists():
            raise RuntimeError(f"whisperx falhou:\n{r.stderr[-1500:]}")
        return json.loads(out.read_text())


def motor_vad(src: Path, language: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "full.wav"
        to_wav(src, wav)
        return whisper_cpp(wav, language, vad=True)


def cut_point(prev: list[dict], nxt: list[dict], lo: float, hi: float) -> float:
    """Instante de troca entre janelas: a maior pausa dentro da sobreposição."""
    cands = []
    for ws in (prev, nxt):
        dentro = [w for w in ws if lo <= w["start"] <= hi]
        cands += [(b["start"] - a["end"], (a["end"] + b["start"]) / 2)
                  for a, b in zip(dentro, dentro[1:])]
    return max(cands)[1] if cands else (lo + hi) / 2


def motor_janela(src: Path, language: str) -> list[dict]:
    total = duration(src)
    merged: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        start, i = 0.0, 0
        while start < total:
            dur = min(WIN, total - start)
            wav = Path(tmp) / f"w{i:03d}.wav"
            to_wav(src, wav, start, dur)
            words = [{"text": w["text"], "start": round(w["start"] + start, 2),
                      "end": round(w["end"] + start, 2)}
                     for w in whisper_cpp(wav, language, vad=False)]
            if not merged:
                merged = words
            else:
                cut = cut_point(merged, words, start, min(start + (WIN - HOP), total))
                merged = [w for w in merged if w["start"] < cut] + \
                         [w for w in words if w["start"] >= cut]
            print(f"  janela {i} ({start:.0f}-{start + dur:.0f}s): {len(words)} palavras",
                  file=sys.stderr)
            start += HOP
            i += 1
    return sorted(merged, key=lambda w: (w["start"], w["end"]))


MOTORES = {"whisperx": motor_whisperx, "vad": motor_vad, "janela": motor_janela}


# -------------------------------------------------------------- verificações

def checar_tempos(words: list[dict], total: float) -> list[str]:
    """Sanidade dos timestamps — legenda fora de sincronia é pior que texto errado."""
    p = []
    if not words:
        return ["transcrição vazia"]
    ruins = [w for w in words if w["end"] <= w["start"]]
    if ruins:
        p.append(f"{len(ruins)} palavras com fim <= início")
    if any(words[i]["start"] > words[i + 1]["start"] for i in range(len(words) - 1)):
        p.append("palavras fora de ordem cronológica")
    longas = [w for w in words if w["end"] - w["start"] > 2.5]
    if longas:
        p.append(f"{len(longas)} palavras com duração > 2.5s (ex: {longas[0]['text']!r} "
                 f"em {longas[0]['start']:.1f}s) — timestamp provavelmente errado")
    if words[-1]["end"] > total + 0.5:
        p.append(f"última palavra ({words[-1]['end']:.1f}s) passa do fim do áudio ({total:.1f}s)")
    fim = total - words[-1]["end"]
    if fim > 5:
        p.append(f"{fim:.1f}s de áudio depois da última palavra — pode ter fala perdida no fim")
    if words[0]["start"] > 5:
        p.append(f"{words[0]['start']:.1f}s de áudio antes da primeira palavra — idem no começo")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--language", default="pt")
    ap.add_argument("--motor", default="whisperx", choices=list(MOTORES))
    ap.add_argument("--verificar", action="store_true",
                    help="roda um segundo motor e mostra as divergências (proteção contra perda de fala)")
    a = ap.parse_args()

    src = Path(a.source)
    total = duration(src)
    print(f"motor: {a.motor} | {total:.1f}s de áudio", file=sys.stderr)
    words = MOTORES[a.motor](src, a.language)
    Path(a.out).write_text(json.dumps({"words": words}, ensure_ascii=False, indent=1))
    print(f"{len(words)} palavras -> {a.out}", file=sys.stderr)
    from leitura import leitura
    Path(a.out).with_suffix(".leitura.txt").write_text(leitura(words))  # versão compacta pra ler (menos token)

    for p in checar_tempos(words, total):
        print(f"  ATENÇÃO (tempo): {p}", file=sys.stderr)

    if a.verificar:
        outro = "vad" if a.motor == "whisperx" else "whisperx"
        print(f"\nverificação cruzada com o motor '{outro}'...", file=sys.stderr)
        w2 = MOTORES[outro](src, a.language)
        alt = Path(a.out).with_suffix(f".{outro}.json")
        alt.write_text(json.dumps({"words": w2}, ensure_ascii=False, indent=1))
        subprocess.run([sys.executable, str(AQUI / "comparar.py"), a.out, str(alt),
                        "--rotulos", f"{a.motor},{outro}"])


if __name__ == "__main__":
    main()
