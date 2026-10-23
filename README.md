# transcrever-ptbr

Transcrição de vídeo/áudio em pt-BR sem perder fala: roda dois motores, compara, aplica glossário de nomes do nicho (`glossario.json`) e valida a sincronia antes de virar legenda.

> Feito pro meu projeto de redes sociais (macOS). Os scripts assumem a pasta `ferramentas/` do projeto e alguns têm caminho absoluto `/Volumes/1TB Mac/...`: ajuste antes de usar em outro lugar.

---

# transcrever — transcrição local confiável (pt-BR)

Por que existe, medição e histórico: `memory/diretrizes-transcricao.md`.
SOP de uso: skill `.claude/skills/transcricao-confiavel/SKILL.md`.

## Instalação (já feita nesta máquina)

```bash
cd ferramentas/transcrever
python3 -m venv venv && ./venv/bin/pip install whisperx
```

O venv é isolado porque o whisperx traz torch 2.8 + pyannote + ctranslate2 — não misturar com o
Python global (que roda yt-dlp/instagram-transcript com `curl_cffi` pinado).

Os motores `vad` e `janela` NÃO usam o venv: chamam o `whisper-cli` global. Variáveis:
`WHISPER_CPP_BIN`, `WHISPER_CPP_MODEL`, `WHISPER_VAD_MODEL` (padrões apontam pra
`/Volumes/1TB Mac/whisper.cpp/`).

## Arquivos

| arquivo | o que faz |
|---|---|
| `transcrever.py` | transcreve; `--motor whisperx\|vad\|janela`, `--verificar` |
| `comparar.py` | diff entre duas transcrições — a proteção contra perda de fala |
| `glossario.py` | aplica `glossario.json` (nomes que o Whisper sempre erra) |
| `leitura.py` | gera `<saída>.leitura.txt` (frase + tempo) — LER este em vez do JSON palavra-a-palavra (~70% menos tokens); `transcrever.py` já chama |
| `glossario.json` | mapa termo-errado → termo-certo do nicho do Pedro |

## Gotchas

- **whisperx roda em CPU nesta máquina.** faster-whisper/CTranslate2 não suporta MPS; o
  `torch.backends.mps.is_available() == True` engana. ~3-5 min pra 2 min de áudio, ~4 GB de RAM.
- Primeira execução baixa ~5 GB pra `~/.cache/huggingface` (large-v3 + alinhador wav2vec2 pt).
- Warning de `torchcodec`/`libavutil` no stderr é inofensivo — ele cai pro decoder de fallback.
- `whisperx` normaliza a pontuação diferente do whisper.cpp; o `comparar.py` já ignora isso.
