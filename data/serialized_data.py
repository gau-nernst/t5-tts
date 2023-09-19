import numpy as np
import torch


class Writer:
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


class Reader:
    def __init__(self, name: str, shape: tuple[int, ...] | None = None) -> None:
        self.index = torch.from_numpy(np.memmap(f"{name}.index", dtype=np.int64, mode="r"))
        self.data = torch.from_numpy(np.memmap(f"{name}.data", dtype=np.int16, mode="r"))
        self.shape = shape

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.data[self.index[index] : self.index[index + 1]]
        if self.shape is not None:
            item = item.view(self.shape)
        return item

    def __len__(self) -> int:
        return len(self.index) - 1
