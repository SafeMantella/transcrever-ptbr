"""Compara duas transcrições do mesmo áudio e mostra onde elas divergem.

É a proteção contra transcrição ruim. Motivo (10/09/2026): procurar "lacuna"
(buraco de tempo entre palavras) NÃO acha a maior parte das perdas — quando o
whisper pula fala, ele costuma emitir timestamps contínuos por cima do buraco.
A perda de ~5s em 52s do vídeo "3 skills" tinha zero buraco de tempo.

O que acha de verdade: rodar duas configurações independentes e diferenciar o
texto. Trecho que aparece numa e não na outra é candidato a perda — aí basta
re-transcrever aquela janela isolada pra decidir quem está certo.

Uso:
    python3 comparar.py a.json b.json [--rotulos "dtw,vad"]
"""
from __future__ import annotations

import argparse, json, re, sys
from difflib import SequenceMatcher
from pathlib import Path


def load(p: str) -> list[dict]:
    d = json.loads(Path(p).read_text())
    return d["words"] if isinstance(d, dict) else d


def norm(t: str) -> str:
    return re.sub(r"[^\wáàâãéêíóôõúçñ]", "", t.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--rotulos", default="A,B")
    args = ap.parse_args()
    la, lb = args.rotulos.split(",")

    A, B = load(args.a), load(args.b)
    ka, kb = [norm(w["text"]) for w in A], [norm(w["text"]) for w in B]

    sm = SequenceMatcher(a=ka, b=kb, autojunk=False)
    print(f"{la}: {len(A)} palavras | {lb}: {len(B)} palavras")
    print(f"similaridade: {sm.ratio():.1%}\n")

    achou = False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # divergência de 1-2 palavras é ruído de pontuação/grafia; ignorar
        if max(i2 - i1, j2 - j1) < 3:
            continue
        achou = True
        t = A[i1]["start"] if i1 < len(A) else (B[j1]["start"] if j1 < len(B) else 0)
        print(f"~{t:6.1f}s  [{tag}]")
        if i2 > i1:
            print(f"   {la}: {' '.join(w['text'] for w in A[i1:i2])}")
        else:
            print(f"   {la}: (nada)")
        if j2 > j1:
            print(f"   {lb}: {' '.join(w['text'] for w in B[j1:j2])}")
        else:
            print(f"   {lb}: (nada)")
        print()

    if not achou:
        print("nenhuma divergência relevante (>=3 palavras).")
    else:
        print("Cada bloco acima: re-transcreva a janela isolada pra decidir quem está certo.")
        print("  ffmpeg -ss <t-2> -t <dur+4> -i video.mp4 -ar 16000 -ac 1 j.wav")
        print("  whisper-cli -m <modelo> -l pt -nt j.wav")


if __name__ == "__main__":
    main()
