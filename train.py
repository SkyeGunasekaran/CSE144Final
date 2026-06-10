"""
CSE 144 Final Project — Transfer Learning with DINOv2
======================================================
Pipeline:
  1. Load & augment training data (100 classes, ~10 imgs/class)
  2. Load pretrained DINOv2 ViT-B/14 from HuggingFace
  3. Attach a linear classifier head for 100 classes
  4. Train with early stopping, save best weights
  5. Run inference on the test set and produce submission.csv

Usage:
  # Training
  python train.py --data_dir /path/to/dataset --mode train

  # Inference only (load saved weights)
  python train.py --data_dir /path/to/dataset --mode infer --checkpoint best_model.pth
"""

import os
import random
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from transformers import AutoModel

# ─────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic=True can slow training. Keep it off for speed unless debugging.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ─────────────────────────────────────────────
# 1.  Dataset
# ─────────────────────────────────────────────
class TrainDataset(Dataset):
    """
    Expects:
        data_dir/
            train/
                0/   1.jpg … 10.jpg
                1/   ...
                99/  ...
    Class label == int(folder_name), so the mapping is always
    deterministic and matches the Kaggle requirement.
    """
 
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
 
        train_dir = Path(data_dir) / "train"
        # Sort numerically so class 0→label 0, class 1→label 1, …
        class_dirs = sorted(
            (p for p in train_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name)
        )
 
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((str(img_path), label))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
 
 
class TestDataset(Dataset):
    """
    Expects:
        data_dir/
            test/
                0.jpg
                1.jpg
                …
                999.jpg
    Returns (image_tensor, image_id) where image_id is the integer stem.
    """
 
    def __init__(self, data_dir: str, transform=None):
        self.transform = transform
        test_dir = Path(data_dir) / "test"
        # Sort numerically by stem (0, 1, 2, …)
        self.samples = sorted(test_dir.glob("*.jpg"), key=lambda p: int(p.stem))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path = self.samples[idx]
        image_id = int(path.stem)
        image = Image.open(str(path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_id


# ─────────────────────────────────────────────
# 2.  Transforms
# ─────────────────────────────────────────────
IMG_SIZE = 224

# DINOv3 was trained with ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Geometric augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # Colour augmentations
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    # Crop after resize for mild scale jitter
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ─────────────────────────────────────────────
# 3.  Model
# ─────────────────────────────────────────────
DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

class DINOv3Classifier(nn.Module):
    """
    DINOv3 ViT-B/16 backbone + single linear head.
    """

    def __init__(self, num_classes: int = 100, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(DINO_MODEL_NAME)
        hidden_dim = self.backbone.config.hidden_size  # 768 for ViT-B

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        # Use the [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)


# ─────────────────────────────────────────────
# 4.  Training utilities
# ─────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler=None,
    history: dict[str, list] | None = None,
    global_step: int = 0,
    epoch: int = 1,
):
    """Train for one epoch and optionally log per-batch loss by global step."""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

        global_step += 1
        batch_loss = loss.item()
        batch_acc = accuracy(logits, labels)

        if history is not None:
            history["step"].append(global_step)
            history["epoch"].append(epoch)
            history["batch"].append(batch_idx)
            history["split"].append("Train")
            history["loss"].append(batch_loss)

        bs = labels.size(0)
        total_loss += batch_loss * bs
        total_acc  += batch_acc * bs
        n          += bs

    return total_loss / n, total_acc / n, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n          += bs

    return total_loss / n, total_acc / n


def plot_loss_history(
    history: dict[str, list],
    epoch_markers: list[tuple[int, int]],
    out_path: str | Path,
    title: str = "Training vs Validation Loss",
):
    """Save a paper-style loss curve over total optimizer steps.

    Train loss is logged every batch. Validation loss is logged once per epoch at
    the cumulative step reached after that epoch. Vertical guide lines mark epoch
    boundaries, with epoch numbers annotated along the top of the plot.
    """
    if not history["step"]:
        print("No training history available; skipping loss plot.")
        return

    # Imported lazily so inference mode does not require plotting dependencies.
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(history)
    out_path = Path(out_path)

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(8.5, 5.25), constrained_layout=True)

    sns.lineplot(
        data=df[df["split"] == "Train"],
        x="step", y="loss",
        estimator=None,
        linewidth=1.8,
        alpha=0.85,
        label="Training loss",
        ax=ax,
    )
    sns.lineplot(
        data=df[df["split"] == "Validation"],
        x="step", y="loss",
        estimator=None,
        marker="o",
        markersize=6,
        linewidth=2.2,
        label="Validation loss",
        ax=ax,
    )

    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.04 * (y_max - y_min)
    for epoch, step in epoch_markers:
        ax.axvline(step, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(
            step, y_text, f"E{epoch}",
            rotation=90,
            va="top", ha="right",
            fontsize=8, alpha=0.75,
        )

    ax.set_title(title, pad=12, weight="bold")
    ax.set_xlabel("Total training steps")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=True, title=None)
    sns.despine(fig=fig, ax=ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot saved to: {out_path}")


# ─────────────────────────────────────────────
# 5.  Main training loop
# ─────────────────────────────────────────────
def train(args):
    set_seed(SEED, deterministic=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Datasets ──────────────────────────────
    full_dataset = TrainDataset(args.data_dir, transform=train_transform)
    n_total  = len(full_dataset)
    n_val    = max(1, int(n_total * args.val_split))
    n_train  = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Val set uses eval (no augmentation) transforms
    val_ds.dataset = TrainDataset(args.data_dir, transform=eval_transform)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    print(f"Train: {n_train} samples  |  Val: {n_val} samples")

    # ── Model ─────────────────────────────────
    model = DINOv3Classifier(num_classes=100, freeze_backbone=True).to(device)
    print(f"Backbone hidden dim: {model.backbone.config.hidden_size}")
    if args.compile and device.type == "cuda":
        # Important: torch.compile returns a new module. Calling it without assignment is a no-op.
        model = torch.compile(model)

    # ── Optimizer / scheduler / loss ──────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path  = args.checkpoint

    history = {
        "step": [],
        "epoch": [],
        "batch": [],
        "split": [],
        "loss": [],
    }
    epoch_markers: list[tuple[int, int]] = []
    global_step = 0

    print("\n=== Training ===")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            history=history, global_step=global_step, epoch=epoch,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["step"].append(global_step)
        history["epoch"].append(epoch)
        history["batch"].append(len(train_loader))
        history["split"].append("Validation")
        history["loss"].append(val_loss)
        epoch_markers.append((epoch, global_step))

        print(
            f"[Head  | Ep {epoch:3d}/{args.epochs}]  "
            f"step={global_step:5d}  "
            f"train loss={tr_loss:.4f}  acc={tr_acc:.3f}  |  "
            f"val loss={val_loss:.4f}  acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model  (val_acc={best_val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("  Early stopping triggered.")
                break

    plot_loss_history(history, epoch_markers, args.loss_plot)

    # Run inference after training
    _run_inference(model, args, device, checkpoint_path)


# ─────────────────────────────────────────────
# 6.  Inference
# ─────────────────────────────────────────────
def _run_inference(model, args, device, checkpoint_path):
    print("\n=== Running inference on test set ===")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_ds     = TestDataset(args.data_dir, transform=eval_transform)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    results: list[tuple[int, int]] = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().tolist()
            for img_id, pred in zip(image_ids.tolist(), preds):
                results.append((img_id, pred))

    # Sort by ID to match sample_submission.csv order
    results.sort(key=lambda x: x[0])

    out_path = Path(args.output_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for img_id, pred in results:
            writer.writerow([f"{img_id}.jpg", pred])

    print(f"Submission saved to: {out_path}  ({len(results)} predictions)")


def infer(args):
    set_seed(SEED, deterministic=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model = DINOv3Classifier(num_classes=100, freeze_backbone=False).to(device)
    _run_inference(model, args, device, args.checkpoint)


# ─────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CSE 144 Transfer Learning")

    p.add_argument("--data_dir",    type=str,   default=".",
                   help="Root directory containing train/ and test/ folders")
    p.add_argument("--mode",        type=str,   default="train",
                   choices=["train", "infer"],
                   help="'train' runs full pipeline; 'infer' loads a checkpoint and predicts")
    p.add_argument("--checkpoint",  type=str,   default="best_model.pth",
                   help="Path to save/load model weights")
    p.add_argument("--output_csv",  type=str,   default="submission.csv",
                   help="Output CSV path for Kaggle submission")
    p.add_argument("--loss_plot",   type=str,   default="loss_curve.png",
                   help="Path to save training-vs-validation loss plot")

    # Data
    p.add_argument("--val_split",   type=float, default=0.15,
                   help="Fraction of training data used for validation")
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile. First epoch may be slower due to compilation.")

    # Training schedule
    p.add_argument("--epochs",   type=int,   default=20,
                   help="Epochs to train")
    p.add_argument("--patience",        type=int,   default=10,
                   help="Early-stopping patience (epochs without val_acc improvement)")

    # Optimiser
    p.add_argument("--lr_head",       type=float, default=1e-3,
                   help="Learning rate for the classification head")
    p.add_argument("--weight_decay",  type=float, default=0)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer(args)