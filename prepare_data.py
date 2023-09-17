import argparse
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sentencepiece as spm
import torch
from tqdm import tqdm

from modelling.encodec import EnCodec


FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")


def get_t5_tokenizer(checkpoint: str, cache: str = "tokenizers") -> spm.SentencePieceProcessor:
    location = "mc4.250000.100extra" if checkpoint.startswith("mt5") else "cc_all.32000.100extra"

    cache_path = Path(cache) / location
    if not cache_path.exists():
        BASE_URL = "https://storage.googleapis.com/t5-data/vocabs"
        cache_path.mkdir(parents=True)

        for filename in ("sentencepiece.model", "sentencepiece.vocab"):
            resp = requests.get(f"{BASE_URL}/{location}/{filename}")
            with open(cache_path / filename, "wb") as f:
                f.write(resp.content)

    return spm.SentencePieceProcessor(str(cache_path / "sentencepiece.model"))


def get_librispeech_meta(data_dir: str, split: str) -> pd.DataFrame:
    sub_dir = Path(data_dir) / split

    filenames = []
    texts = []
    speaker_ids = []
    chapter_ids = []

    for speaker_id in sub_dir.iterdir():
        for chapter_id in speaker_id.iterdir():
            transcript = f"{speaker_id.name}-{chapter_id.name}.trans.txt"

            lines = [line.rstrip() for line in open(chapter_id / transcript)]
            speaker_ids.extend([speaker_id.name] * len(lines))
            chapter_ids.extend([chapter_id.name] * len(lines))
            for line in lines:
                filename, text = line.split(maxsplit=1)
                filenames.append(f"{speaker_id.name}/{chapter_id.name}/{filename}.flac")
                texts.append(text.lower())

    return pd.DataFrame(dict(filename=filenames, text=texts, speaker_id=speaker_ids, chapter_id=chapter_ids))


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    cmd = f"{FFMPEG_PATH} -i {path} -ar {sample_rate} -ac 1 -f s32le -"
    proc = subprocess.run(shlex.split(cmd), capture_output=True)

    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    return torch.frombuffer(proc.stdout, dtype=torch.int32) / 2_147_483_648


class SerializedDataWriter:
    def __init__(self, name: str) -> None:
        self.index = open(f"{name}.index", "wb")
        self.data = open(f"{name}.data", "wb")
        self.pos = np.array(0, dtype=np.int64)

        self.index.write(self.pos.tobytes())

    def write(self, item: np.ndarray) -> None:
        self.data.write(item.tobytes())
        self.pos += item.size
        self.index.write(self.pos.tobytes())

    def close(self) -> None:
        self.index.close()
        self.data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5_model", required=True)
    parser.add_argument("--encodec_model", required=True)
    parser.add_argument("--n_quantizers", type=int, default=4)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)

    args = parser.parse_args()

    encodec = EnCodec.from_facebook(args.encodec_model, pretrained=True, decoder=False).eval()

    meta = get_librispeech_meta(args.data_dir, args.split)

    tokenizer = get_t5_tokenizer(args.t5_model)
    all_text_ids = tokenizer.Encode(meta["text"].to_list(), add_eos=True)

    text_writer = SerializedDataWriter("text")
    for text_ids in all_text_ids:
        text_ids = np.array(text_ids, dtype=np.int16)  # vocab size is 32,000
        text_writer.write(text_ids)
    text_writer.close()

    audio_writer = SerializedDataWriter("audio")

    for filename in tqdm(meta["filename"]):
        audio = load_audio(args.data_dir / args.split / filename, 24_000)
        audio_ids, scale = encodec.encode(audio.view(1, 1, -1), args.n_quantizers)
        audio_ids = audio_ids.squeeze()  # (n_quantizers, length)

        audio_ids = audio_ids.cpu().to(torch.int16).numpy()  # codebook size is 1024
        audio_writer.write(audio_ids)

    audio_writer.close()
