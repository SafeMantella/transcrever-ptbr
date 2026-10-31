"""Transcrição local confiável (pt-BR) via VoiceStudio — 3 motores + verificação cruzada.

Padrão: --motor whisperx (transcreve o texto num passe e alinha palavra por
palavra com Wav2vec2). Foi o único dos quatro caminhos testados em 10/09/2026
que devolveu o áudio inteiro sem perder fala. Ver memory/diretrizes-transcricao.md
pra medição completa e pra quando usar cada motor.

Os motores rodam no VoiceStudio (servidor local, http://127.0.0.1:3900) — ver
voicestudio.py pro mapa motor -> engine. Nenhum modelo nem venv aqui.

    python3 transcrever.py video.mp4 -o saida.json                # whisperx (padrão)
    python3 transcrever.py video.mp4 -o saida.json --motor janela # faster-whisper deslizante
    python3 transcrever.py video.mp4 -o saida.json --motor vad    # faster-whisper + Silero VAD
    python3 transcrever.py video.mp4 -o saida.json --verificar    # roda 2 motores e diffa

Saída: {"words": [{"text", "start", "end"}, ...]} — mesmo formato que o resto do
pipeline (pack_transcripts.py, render.py, build de legenda) já espera.
"""
from __future__ import annotations

import argparse, json, subprocess, sys, tempfile
from pathlib import Path

from voicestudio import motor_ativo, transcrever_wav

AQUI = Path(__file__).resolve().parent

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


def avisar_faltando(words: list[dict], texto: str) -> None:
    """O VoiceStudio descarta palavra que o alinhador deixou sem tempo (número, símbolo)."""
    falta = len(texto.split()) - len(words)
    if falta > 0:
        print(f"  ATENÇÃO (texto): {falta} palavra(s) do texto sem timestamp ficaram de fora — "
              "conferir número/sigla no comparar.py", file=sys.stderr)


def motor_arquivo(src: Path, language: str, motor: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp, motor_ativo(motor):
        wav = Path(tmp) / "full.wav"
        to_wav(src, wav)
        words, texto = transcrever_wav(wav, language)
    avisar_faltando(words, texto)
    return words


# ------------------------------------------------------------------- motores

def motor_whisperx(src: Path, language: str) -> list[dict]:
    """Melhor resultado medido."""
    return motor_arquivo(src, language, "whisperx")


def motor_vad(src: Path, language: str) -> list[dict]:
    return motor_arquivo(src, language, "vad")


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
    with tempfile.TemporaryDirectory() as tmp, motor_ativo("vad"):
        start, i = 0.0, 0
        while start < total:
            dur = min(WIN, total - start)
            wav = Path(tmp) / f"w{i:03d}.wav"
            to_wav(src, wav, start, dur)
            words = [{"text": w["text"], "start": round(w["start"] + start, 2),
                      "end": round(w["end"] + start, 2)}
                     for w in transcrever_wav(wav, language)[0]]
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
