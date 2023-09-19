import os
import shlex
import subprocess

import torch


FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    cmd = f"{FFMPEG_PATH} -i {path} -ar {sample_rate} -ac 1 -f s32le -"
    proc = subprocess.run(shlex.split(cmd), capture_output=True)

    if proc.returncode:
        raise RuntimeError(proc.stderr.decode())
    return torch.frombuffer(proc.stdout, dtype=torch.int32) / 2_147_483_648
