"""Cliente mínimo do VoiceStudio (servidor local de ASR) — só stdlib.

Cada motor do transcrever.py é um motor de ASR do VoiceStudio. Os dois usam o
mesmo modelo (faster-whisper large-v3), então a verificação cruzada continua
sendo "dois decodificadores diferentes", não dois modelos:

    whisperx -> whisperx        (texto + alinhamento palavra a palavra com Wav2vec2)
    vad      -> faster-whisper  (Silero VAD embutido, sem alinhador)

O motor ativo no VoiceStudio é global (vale pra UI e pra outras ferramentas), então
`motor_ativo()` troca só durante a chamada e devolve o anterior no fim.
URL em VOICESTUDIO_URL (padrão http://127.0.0.1:3900).
"""
from __future__ import annotations

import json, os, sys, urllib.error, urllib.request, uuid
from contextlib import contextmanager
from pathlib import Path

URL = os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900").rstrip("/")
ENGINES = {"whisperx": "whisperx", "vad": "faster-whisper"}
TIMEOUT = 3600  # arquivo longo em CPU demora; quem trabalha é o servidor


def _req(method: str, path: str, body: bytes | None = None, ctype: str = "application/json") -> dict:
    r = urllib.request.Request(f"{URL}{path}", data=body, method=method,
                               headers={"Content-Type": ctype} if body else {})
    try:
        with urllib.request.urlopen(r, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"VoiceStudio {path} falhou ({e.code}): {e.read()[-800:].decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"VoiceStudio fora do ar em {URL} ({e.reason}) — subir antes (memory/voicestudio-audio-local.md)")


def ativo() -> str:
    return _req("GET", "/engines/asr")["active"]


def _selecionar(engine: str) -> None:
    _req("POST", "/engines/select", json.dumps({"family": "asr", "backend_id": engine}).encode())


@contextmanager
def motor_ativo(motor: str):
    alvo, antes = ENGINES[motor], ativo()
    if alvo != antes:
        _selecionar(alvo)
    try:
        yield
    finally:
        if alvo != antes:
            _selecionar(antes)


def transcrever_wav(wav: Path, language: str = "pt") -> tuple[list[dict], str]:
    """wav -> ([{text, start, end}], texto corrido) com o motor que estiver ativo."""
    b = uuid.uuid4().hex
    campos = [("model", "whisper-1"), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "word"), ("language", language)]
    corpo = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                     for k, v in campos)
    corpo += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{wav.name}"\r\n'
              f"Content-Type: audio/wav\r\n\r\n").encode() + wav.read_bytes() + f"\r\n--{b}--\r\n".encode()
    raw = _req("POST", "/v1/audio/transcriptions", corpo, f"multipart/form-data; boundary={b}")
    if "words" not in raw:
        raise RuntimeError("VoiceStudio não devolveu timestamp por palavra — motor ativo não suporta")
    words = [{"text": w["word"].strip(), "start": round(w["start"], 2), "end": round(w["end"], 2)}
             for w in raw["words"] if w["word"].strip()]
    return words, raw.get("text", "")
