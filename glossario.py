"""Aplica o glossário do projeto numa transcrição (nomes que o Whisper sempre erra).

Uso: python3 glossario.py entrada.json -o saida.json [-g glossario.json]

Chave de várias palavras ("andre carpat skills") funde as palavras numa só,
herdando o start da primeira e o end da última — o timestamp continua correto.
Preserva a pontuação que estava grudada na última palavra do trecho.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path


def core(t: str) -> str:
    return t.strip().strip(".,!?;:…\"'()").lower()


def tail(t: str) -> str:
    m = re.search(r"[.,!?;:…\"')]+$", t.strip())
    return m.group(0) if m else ""


def apply(words: list[dict], gloss: dict[str, str]) -> tuple[list[dict], list[str]]:
    frases = sorted([(k.split(), v) for k, v in gloss.items() if not k.startswith("_")],
                    key=lambda kv: -len(kv[0]))
    out, log, i = [], [], 0
    while i < len(words):
        casou = False
        for chave, novo in frases:
            n = len(chave)
            if i + n > len(words):
                continue
            if [core(w["text"]) for w in words[i:i + n]] == chave:
                antes = " ".join(w["text"] for w in words[i:i + n])
                out.append({"text": novo + tail(words[i + n - 1]["text"]),
                            "start": words[i]["start"], "end": words[i + n - 1]["end"]})
                log.append(f"{antes} -> {novo}")
                i += n
                casou = True
                break
        if not casou:
            out.append(words[i])
            i += 1
    return out, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada"); ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-g", "--glossario", default=str(Path(__file__).parent / "glossario.json"))
    a = ap.parse_args()
    d = json.loads(Path(a.entrada).read_text())
    words = d["words"] if isinstance(d, dict) else d
    novos, log = apply(words, json.loads(Path(a.glossario).read_text()))
    Path(a.out).write_text(json.dumps({"words": novos}, ensure_ascii=False, indent=1))
    from collections import Counter
    for troca, n in Counter(log).most_common():
        print(f"  {n}x  {troca}")
    print(f"{len(novos)} palavras -> {a.out}")


if __name__ == "__main__":
    main()
