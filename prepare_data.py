import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import serialized_data
from data.io import load_audio
from modelling import EnCodec, T5Model


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


class AudioDataset(Dataset):
    def __init__(self, data_dir: Path, files: list[str]) -> None:
        self.data_dir = data_dir
        self.files = files

    def __getitem__(self, index) -> torch.Tensor:
        return load_audio(self.data_dir / self.files[index], 16_000)

    def __len__(self) -> int:
        return len(self.files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5_model", default="flan_t5")
    parser.add_argument("--encodec_model", default="24khz")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encodec = EnCodec.from_facebook(args.encodec_model, pretrained=True, decoder=False).eval()
    encodec.to(args.device)

    meta = get_librispeech_meta(args.data_dir, args.split)

    tokenizer = T5Model.get_tokenizer(args.t5_model)
    all_text_ids = tokenizer.Encode(meta["text"].to_list(), add_eos=True)

    text_writer = serialized_data.Writer("text")
    for text_ids in all_text_ids:
        text_ids = np.array(text_ids, dtype=np.int16)  # vocab size is 32,000
        text_writer.write(text_ids)
    text_writer.close()

    audio_writer = serialized_data.Writer("audio")

    # use PyTorch's DataLoader to load data in a separate process
    ds = AudioDataset(args.data_dir / args.split, meta["filename"].to_list())
    dloader = DataLoader(ds, num_workers=1)

    for audio in tqdm(dloader):
        audio_ids, scale = encodec.encode(audio.view(1, 1, -1).to(args.device))
        audio_ids = audio_ids.squeeze()  # (n_quantizers, length)

        audio_ids = audio_ids.cpu().numpy().astype(np.int16)  # codebook size is 1024
        audio_writer.write(audio_ids)

    audio_writer.close()
