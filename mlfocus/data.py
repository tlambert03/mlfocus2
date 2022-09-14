from typing import Any, Callable, Optional
from torch.utils.data import Dataset
from torchvision import transforms as tf

from pathlib import Path
import zarr
import torch
import re


class SheetAlign(Dataset):
    """

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = [
        "0 - beads",
        "1 - diffuse",
        "2 - mt",
    ]

    def __init__(
        self,
        root: str = "~/Downloads/mlfocus/zarr",
        train: bool = True,
        transforms: Optional[Callable] = None,
        patch_size: int = 64,
        download: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.patch_size = patch_size
        if transforms is None:
            transforms = tf.Compose([tf.ToTensor(), tf.RandomHorizontalFlip()])
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.arrays, self.keys, self.targets = self._load_data()
        self.classes = []
        for a in self.arrays:
            clsname = re.split("\d", Path(a.store.path).stem)[0]
            self.classes.extend([clsname] * len(a))

    def __len__(self):
        return len(self.targets)

    def _check_exists(self) -> bool:
        return self.root.exists()

    def _load_data(self):
        arrays = [zarr.open(i)["raw"] for i in sorted(self.root.glob("*zarr"))]
        keys = [(i, x) for i, a in enumerate(arrays) for x in range(len(a))]
        targets = [(s - 25) / 10 for _, s in keys]
        return arrays, keys, targets

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        (file, focus), target = self.keys[index], self.targets[index]

        img = self.arrays[file][focus]

        img = self._random_crop(img, self.patch_size)

        img = img.astype("float32")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def _random_crop(self, img: zarr.Array, size, min_intensity=190):

        z, h, w = img.shape
        tz, th, tw = (size,) * 3

        if z + 1 < tz or h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required crop size {(tz, th, tw)} is larger then input image size {(z, h, w)}"
            )

        deep = int(torch.randint(0, z - tz + 1, size=(1,)).item())
        top = int(torch.randint(0, h - th + 1, size=(1,)).item())
        left = int(torch.randint(0, w - tw + 1, size=(1,)).item())

        crop = img[deep : deep + tz, top : top + th, left : left + tw]

        while crop.max() < min_intensity:
            crop = self._random_crop(img, size, min_intensity)

        return crop
