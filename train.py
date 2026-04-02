"""
Punto di ingresso per il training.

Uso:
    python train.py                     # usa configs/base.yaml
    python train.py model.name=lstm     # override singolo parametro
"""


import torch
torch.set_float32_matmul_precision('high')
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers   import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from data.datamodule     import EGTEADataModule
from models.cnn1d        import CNN1DModel
from models.lstm         import LSTMModel
from training.module     import TemporalSegmentationModule


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Supporto override tipo "model.name=lstm"
    for ov in overrides:
        key, val = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Cast automatico
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        d[keys[-1]] = val
    return cfg


def build_model(cfg: dict):
    name = cfg["model"]["name"]
    kwargs = dict(
        feat_dim    = cfg["model"]["feat_dim"],
        num_classes = cfg["model"]["num_classes"],
        hidden      = cfg["model"]["hidden"],
    )
    if name == "cnn1d":
        return CNN1DModel(**kwargs)
    elif name == "lstm":
        return LSTMModel(
            **kwargs,
            n_layers      = cfg["model"].get("n_layers", 2),
            bidirectional = cfg["model"].get("bidirectional", True),
        )
    else:
        raise ValueError(f"Modello non riconosciuto: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    # --- Data ---
    datamodule = EGTEADataModule(**cfg["data"])

    # --- Model ---
    model     = build_model(cfg)
    lit_model = TemporalSegmentationModule(
        model          = model,
        num_classes    = cfg["model"]["num_classes"],
        lr             = cfg["training"]["lr"],
        weight_decay   = cfg["training"]["weight_decay"],
        label_smoothing= cfg["training"]["label_smoothing"],
    )

    # --- Logger W&B ---
    logger = WandbLogger(
        project  = cfg["wandb"]["project"],
        name     = cfg["wandb"]["name"],
        log_model= True,
        config   = cfg,
    )
    logger.watch(model, log="gradients", log_freq=50)

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            monitor    = "val/mIoU_epoch",
            mode       = "max",
            save_top_k = 3,
            filename   = "{epoch:02d}-{val/mIoU_epoch:.3f}",
        ),
        EarlyStopping(
            monitor  = "val/mIoU_epoch",
            patience = 5,
            mode     = "max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs        = cfg["training"]["max_epochs"],
        logger            = logger,
        callbacks         = callbacks,
        accelerator       = "cuda",
        log_every_n_steps = 10,
        num_sanity_val_steps = 0,   
        gradient_clip_val    = 1.0,   
    )

    trainer.fit(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()