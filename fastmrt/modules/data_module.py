import os.path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler
from fastmrt.data.dataset import SliceDataset
from fastmrt.pretrain.mri_data import SliceDataset as pt_SliceDataset
from fastmrt.pretrain.imagenet_data import ImagenetDataset as pt_ImagenetDataset
from pathlib import Path
from typing import Callable, Any


class FastmrtDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root: Path,
            only_source: bool,
            train_transform: Callable,
            val_transform: Callable,
            test_transform: Callable,
            batch_size: int = 16,
            dataset_type: str = '2D',
            collate_fn=None,
            work_init_fn: Any = None,
            generator: Any = None,
            workers: int=0,
    ):
        super(FastmrtDataModule, self).__init__()
        self.root = root
        self.only_source = only_source
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.collate_fn = collate_fn
        self.work_init_fn = work_init_fn
        self.generator = generator
        self.workers = workers

    def train_dataloader(self):
        return self._create_dataloader(stage='train', transform=self.train_transform)

    def val_dataloader(self):
        return self._create_dataloader(stage='val', transform=self.val_transform)

    def test_dataloader(self):
        return self._create_dataloader(stage='test', transform=self.test_transform)

    def _create_dataloader(
            self,
            stage: str = 'train',
            transform: Callable = None,
    ) -> DataLoader[Any]:
        data_path = [os.path.join(sub_data_path, stage, self._sub_folder(stage)) for sub_data_path in self.root]
        shuffle = True

        # choose transform depend on stage
        if stage == 'train':
            if self.collate_fn is None:
                transform = self.train_transform
            else:
                transform = None
        elif stage == 'val':
            transform = self.val_transform
            shuffle = False
        elif stage == 'test':
            transform = self.test_transform
            shuffle = False

        # load dataset
        if self.dataset_type == '2D':
            dataset = SliceDataset(root=data_path, transform=transform)
        elif self.dataset_type == 'PT':
            # dataset = pt_SliceDataset(root=data_path, challenge="singlecoil", transform=transform)
            dataset = pt_ImagenetDataset(root=data_path, transform=transform)
        else:
            raise ValueError("``dataset_type`` must be one of ``2D``, ``3D`` and ``T-3D``")

        if stage == "train":
            collate_fn = self.collate_fn
        else:
            collate_fn = None

        # generate dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            drop_last=True,
            collate_fn=collate_fn,
            worker_init_fn=self.work_init_fn,
            generator=self.generator
        )

        return dataloader
    
    def _sub_folder(self, stage: str):
        if self.only_source is True and stage == 'train':
            return 'source'
        else:
            return ''


class FastmrtPretrainDataModule(pl.LightningDataModule):
    pass
