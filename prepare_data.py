import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from modelling.encodec import EnCodec
from modelling.t5 import T5Model
from utils import load_audio


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
    parser.add_argument("--t5_model", default="flan_t5")
    parser.add_argument("--encodec_model", default="24khz")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    encodec = EnCodec.from_facebook(args.encodec_model, pretrained=True, decoder=False).eval()

    meta = get_librispeech_meta(args.data_dir, args.split)

    tokenizer = T5Model.get_tokenizer(args.t5_model)
    all_text_ids = tokenizer.Encode(meta["text"].to_list(), add_eos=True)

    text_writer = SerializedDataWriter("text")
    for text_ids in all_text_ids:
        text_ids = np.array(text_ids, dtype=np.int16)  # vocab size is 32,000
        text_writer.write(text_ids)
    text_writer.close()

    audio_writer = SerializedDataWriter("audio")

    for filename in tqdm(meta["filename"]):
        audio = load_audio(args.data_dir / args.split / filename, 24_000)
        audio_ids, scale = encodec.encode(audio.view(1, 1, -1))
        audio_ids = audio_ids.squeeze()  # (n_quantizers, length)

        audio_ids = audio_ids.cpu().numpy().astype(np.int16)  # codebook size is 1024
        audio_writer.write(audio_ids)

    audio_writer.close()
