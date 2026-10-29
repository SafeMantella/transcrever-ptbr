# transcrever-ptbr

**Transcrição local de vídeo/áudio em português do Brasil que não perde fala.** Roda no seu computador (sem API, sem custo por minuto), gera o texto palavra por palavra com tempo, corrige os nomes que o Whisper sempre erra e confere uma transcrição contra a outra pra pegar trecho sumido antes de virar legenda.

```text
video.mp4 ──► transcrever.py ──► saida.json          (palavra, início, fim)
                              └► saida.leitura.txt   (uma linha por frase, com tempo; barato de ler)
saida.json ──► glossario.py ──► saida corrigida ("cloud code" -> "Claude Code")
a.json + b.json ──► comparar.py ──► onde as duas transcrições divergem = candidato a fala perdida
```

## Por que existe

Em teste com um vídeo de ~50 s, o Whisper pulou ~5 s de fala **sem deixar buraco nos timestamps**: ele emite tempos contínuos por cima da parte que perdeu. Procurar "lacuna" não acha isso. O que acha é rodar duas configurações independentes e comparar o texto: trecho que aparece numa e não na outra é suspeito, e basta re-transcrever aquela janela isolada pra decidir qual está certa.

Dos quatro caminhos testados em 10/09/2026, só o `whisperx` (passe único + alinhamento Wav2vec2 palavra a palavra) devolveu o áudio inteiro. Por isso é o motor padrão.

## O que tem aqui

| arquivo | o que faz |
|---|---|
| `transcrever.py` | transcreve; `--motor whisperx\|vad\|janela` e `--verificar` (roda dois motores e compara) |
| `comparar.py` | diff entre duas transcrições do mesmo áudio; a proteção contra perda de fala |
| `glossario.py` + `glossario.json` | troca termos que o Whisper erra por termos certos; chave de várias palavras vira uma palavra só, com o tempo da primeira ao fim da última |
| `leitura.py` | gera `<saída>.leitura.txt` (frase + tempo). Ler esse arquivo no lugar do JSON palavra a palavra custa ~70% menos token. `transcrever.py` já chama |

## Instalação

Requisitos: Python 3.10+, `ffmpeg` no PATH, ~5 GB livres (a primeira execução baixa o large-v3 e o alinhador pt pra `~/.cache/huggingface`). Testado em macOS Apple Silicon.

```bash
git clone https://github.com/SafeMantella/transcrever-ptbr.git
cd transcrever-ptbr
python3 -m venv venv && ./venv/bin/pip install whisperx
```

O `venv/` fica **dentro da pasta** porque o `transcrever.py` o procura ali (`venv/bin/python`). Ele é isolado de propósito: o whisperx traz torch, pyannote e ctranslate2, e misturar com o Python global quebra outras ferramentas.

Os motores `vad` e `janela` não usam o venv: chamam o [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) (`whisper-cli`) que você já tiver. Aponte pra ele com variáveis de ambiente:

```bash
export WHISPER_CPP_BIN=/caminho/whisper-cli
export WHISPER_CPP_MODEL=/caminho/ggml-large-v3.bin
export WHISPER_VAD_MODEL=/caminho/ggml-silero-v5.1.2.bin   # só o motor vad
```

Os padrões do código apontam pra `/Volumes/1TB Mac/whisper.cpp/...` (a máquina do autor): sem essas variáveis, só o `whisperx` funciona.

## Uso

```bash
python3 transcrever.py video.mp4 -o saida.json                  # whisperx (padrão)
python3 transcrever.py video.mp4 -o saida.json --motor vad      # whisper.cpp + VAD
python3 transcrever.py video.mp4 -o saida.json --motor janela   # whisper.cpp com janela deslizante (24 s, 6 s de sobreposição)
python3 transcrever.py video.mp4 -o saida.json --verificar      # dois motores + diff
python3 glossario.py saida.json -o saida.corrigida.json [-g glossario.json]
python3 comparar.py a.json b.json --rotulos "whisperx,vad"
```

Saída: `{"words": [{"text", "start", "end"}, ...]}`, tempos em segundos. Esse formato é o que o [video-cut-pipeline](https://github.com/SafeMantella/video-cut-pipeline) lê.

### Qual motor usar

| motor | quando | custo |
|---|---|---|
| `whisperx` | padrão; texto completo com tempo por palavra | CPU, ~3–5 min pra 2 min de áudio, ~4 GB de RAM |
| `vad` | segunda opinião; pula silêncio com detecção de voz | rápido (whisper.cpp) |
| `janela` | quando o áudio tem trechos que os outros engolem | médio |

Pra qualquer legenda que vai ao ar, use `--verificar` e leia o diff.

### Glossário

`glossario.json` é um mapa `"o que sai": "o certo"`, ex.: `"cloud code": "Claude Code"`, `"chat gpt": "ChatGPT"`. O que vem no repo é o vocabulário do autor (IA e programação); troque pelo do seu nicho.

## Gotchas

- **O whisperx roda em CPU**, não em GPU do Mac: faster-whisper/CTranslate2 não suporta MPS, e `torch.backends.mps.is_available() == True` engana.
- Aviso de `torchcodec`/`libavutil` no stderr é inofensivo (ele cai pro decoder de fallback).
- O whisperx normaliza pontuação diferente do whisper.cpp; o `comparar.py` já ignora isso.
- Fragmento de 3 palavras ou menos sai instável em qualquer motor; confira o contexto.
- Em silêncio, o Whisper alucina texto ("Tchau, tchau"): não transcreva trecho de pausa.

## Repos relacionados

Parte do kit do [@SafeMantella](https://github.com/SafeMantella): usado pelo [video-cut-pipeline](https://github.com/SafeMantella/video-cut-pipeline); o `.leitura.txt` é o que o [token-kit](https://github.com/SafeMantella/token-kit) prefere ler. Áudio limpo transcreve melhor: veja [Adobe-Podcast-for-Claude-Code](https://github.com/SafeMantella/Adobe-Podcast-for-Claude-Code).
