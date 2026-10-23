#!/usr/bin/env python3
"""Versão compacta pra LER (texto por frase com tempo), no lugar do JSON palavra-a-palavra (~70% menos tokens).
Uso: leitura.py transcricao.json  -> escreve transcricao.leitura.txt ; o JSON continua sendo a fonte pros scripts."""
import json, sys
from pathlib import Path

def leitura(words, pausa=0.7):
    linhas, atual, t0 = [], [], 0.0
    for i, w in enumerate(words):
        if not atual: t0 = w["start"]
        atual.append(w["text"])
        prox = words[i + 1]["start"] if i + 1 < len(words) else None
        if prox is None or prox - w["end"] > pausa or w["text"][-1:] in ".?!":
            linhas.append(f"[{t0:.1f}] {' '.join(atual)}"); atual = []
    return "\n".join(linhas)

if __name__ == "__main__":
    p = Path(sys.argv[1]); out = p.with_suffix(".leitura.txt")
    out.write_text(leitura(json.loads(p.read_text())["words"])); print(out)
