import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import EGTEADataset


class EGTEADataModule(pl.LightningDataModule):
    def __init__(
        self,
        egtea_root: str,           # es. "D:/egtea"
        annotation_dir: str,       # es. "./action_annotation"
        split: int = 1,            # 1, 2 o 3
        use_flow: bool = False,
        batch_size: int = 8,
        seq_len: int = 256,  
        feat_dim: int = 1024,
        num_workers: int = 0,      # 0 su Windows per evitare problemi multiprocessing
    ):
        super().__init__()
        self.save_hyperparameters()

    def _lmdb_path(self, modality: str) -> str:
        s = self.hparams.split
        name = f"TSN-C_3_egtea_action_CE_s{s}_{modality}_model_best_fcfull_hd"
        return f"{self.hparams.egtea_root}/{name}"

    def setup(self, stage=None):
        hp = self.hparams
        common = dict(
            lmdb_rgb_path  = self._lmdb_path("rgb"),
            annotation_dir = hp.annotation_dir,
            seq_len        = hp.seq_len,
            feat_dim       = hp.feat_dim,
            use_flow       = hp.use_flow,
            lmdb_flow_path = self._lmdb_path("flow") if hp.use_flow else None,
        )
        self.train_ds = EGTEADataset(split_file=f"train_split{hp.split}.txt", **common)
        self.val_ds   = EGTEADataset(split_file=f"test_split{hp.split}.txt",  **common)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.hparams.batch_size,
            shuffle     = True,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.num_workers > 0,
        )