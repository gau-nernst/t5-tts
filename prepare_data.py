import argparse
from pathlib import Path

import pandas as pd
import requests
import sentencepiece as spm

from modelling.encodec import EnCodec


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5_model", required=True)
    parser.add_argument("--encodec_model", required=True)
    parser.add_argument("--n_quantizers", type=int, default=4)
    parser.add_argument("--data_dir", required=True)

    args = parser.parse_args()

    tokenizer = get_t5_tokenizer(args.t5_model)
    encodec = EnCodec.from_facebook(args.encodec_model, pretrained=True)
