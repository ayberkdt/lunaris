---
name: pytorch-lightning
description: Deep learning framework (PyTorch Lightning). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines, callbacks, logging (W&B, TensorBoard), distributed training (DDP, FSDP, DeepSpeed), for scalable neural network training.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

## Overview
PyTorch Lightning is a deep learning framework that organizes PyTorch code to eliminate boilerplate while maintaining full flexibility. Automate training workflows, multi-device orchestration, and implement best practices for neural network training and scaling across multiple GPUs/TPUs.

## When to Use This Skill
- Building, training, or deploying neural networks using PyTorch Lightning
- Organizing PyTorch code into LightningModules
- Configuring Trainers for multi-GPU/TPU training
- Implementing data pipelines with LightningDataModules
- Working with callbacks, logging, and distributed training strategies (DDP, FSDP, DeepSpeed)
- Structuring the HNN (Sobolev-Trained Lunar Residual Potential Surrogate) surrogate gravity model professionally

## Core Components

### 1. LightningModule — Model Definition
Organize PyTorch models into six logical sections:

1. **Initialization** - `__init__()` and `setup()`
2. **Training Loop** - `training_step(batch, batch_idx)`
3. **Validation Loop** - `validation_step(batch, batch_idx)`
4. **Test Loop** - `test_step(batch, batch_idx)`
5. **Prediction** - `predict_step(batch, batch_idx)`
6. **Optimizer Configuration** - `configure_optimizers()`

```python
import lightning as L
import torch
import torch.nn.functional as F

class MyModel(L.LightningModule):
    def __init__(self, hidden_size=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Save all __init__ args
        self.model = YourNetwork(hidden_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.model(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.model(x), y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self.model(x), y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
```

### 2. Trainer — Training Automation
```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

# Basic training
trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(model, train_loader, val_loader)

# Full configuration
trainer = L.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=2,                          # Number of GPUs
    strategy="ddp",                     # DDP for multi-GPU
    precision="16-mixed",               # AMP (2x speedup)
    gradient_clip_val=1.0,              # Gradient clipping
    accumulate_grad_batches=4,          # Gradient accumulation
    log_every_n_steps=10,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ],
    logger=TensorBoardLogger("logs/"),
)
trainer.fit(model, datamodule=dm)
```

### 3. LightningDataModule — Data Pipeline
```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Download data — called once, single process
        pass

    def setup(self, stage=None):
        # Create datasets — called per GPU
        if stage == "fit" or stage is None:
            self.train_dataset = MyDataset(self.hparams.data_dir, split="train")
            self.val_dataset = MyDataset(self.hparams.data_dir, split="val")
        if stage == "test" or stage is None:
            self.test_dataset = MyDataset(self.hparams.data_dir, split="test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                         num_workers=self.hparams.num_workers)
```

### 4. Callbacks
```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,    # Save best/latest models
    EarlyStopping,      # Stop when metrics plateau
    LearningRateMonitor # Track LR scheduler changes
)

checkpoint_cb = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="model-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min",
    verbose=True
)
```

### 5. Logging
```python
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# TensorBoard (default)
logger = TensorBoardLogger("tb_logs/", name="my_model")

# Weights & Biases
logger = WandbLogger(project="lunar-simulation", name="hnn-v2")

# In LightningModule — log anything
self.log("train_loss", loss)
self.log("val_loss", loss, on_step=False, on_epoch=True)
self.log_dict({"loss": loss, "lr": lr, "grad_norm": grad_norm})
```

### 6. Distributed Training
Choose strategy based on model size:
- **DDP** — For models < 500M parameters
- **FSDP** — For models 500M+ parameters
- **DeepSpeed** — For cutting-edge features

```python
# DDP (most common)
trainer = L.Trainer(accelerator="gpu", devices=4, strategy="ddp")

# FSDP for large models
trainer = L.Trainer(accelerator="gpu", devices=4, strategy="fsdp")

# Mixed precision
trainer = L.Trainer(precision="16-mixed")  # AMP
trainer = L.Trainer(precision="bf16-mixed")  # BF16 (better for A100+)
```

## Quick Training Pipeline
```python
import lightning as L
from torch.utils.data import DataLoader

# 1. Define model
model = MyModel(hidden_size=128, lr=1e-3)

# 2. Prepare data
dm = MyDataModule(data_dir="data/", batch_size=64)

# 3. Configure trainer
trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",  # Auto-detect GPU/CPU
    callbacks=[ModelCheckpoint(monitor="val_loss"), EarlyStopping(monitor="val_loss")],
)

# 4. Train
trainer.fit(model, datamodule=dm)

# 5. Test
trainer.test(model, datamodule=dm)

# 6. Inference
predictions = trainer.predict(model, datamodule=dm)
```

## Best Practices
- **Device agnostic**: Use `self.device` instead of `.cuda()`
- **Hyperparameter saving**: Always call `self.save_hyperparameters()` in `__init__()`
- **Metric logging**: Use `self.log()` for automatic aggregation across devices
- **Reproducibility**: Use `L.seed_everything(42)` and `Trainer(deterministic=True)`
- **Debugging**: Use `Trainer(fast_dev_run=True)` to test with 1 batch
- **Progress**: Use `prog_bar=True` in `self.log()` for important metrics

## Installation
```bash
uv pip install lightning torch torchvision
```
