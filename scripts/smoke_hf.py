"""Load an HF Whisper checkpoint on MPS in float16 and transcribe clips in Danish.

    .venv/bin/python -I scripts/smoke_hf.py models/hf/whisper-large-v3 clip1.wav ...

Beam 5, language="da", task="transcribe", to sit next to crispasr's `-bs 5` output.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_wav(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    assert rate == 16000 and audio.ndim == 1, path
    return audio


def main():
    model_dir = Path(sys.argv[1])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16).to(device)
    model.eval()
    print(json.dumps({"device": device, "dtype": str(model.dtype),
                      "load_seconds": round(time.perf_counter() - t0, 2)}))

    for wav in map(Path, sys.argv[2:]):
        audio = load_wav(wav)
        feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(device, torch.float16)
        t0 = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(feats, language="da", task="transcribe", num_beams=5)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(json.dumps({"clip": wav.name, "seconds": round(time.perf_counter() - t0, 2),
                          "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
