import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

def parse_args():
    p = argparse.ArgumentParser(description="Train Chihuahua vs Muffin classifier (PyTorch, CUDA-ready)")
    p.add_argument(
        "--data-dir",
        type=str,
        default="./data", 
        help="Root data directory (with train/val subfolders)"
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-out", type=str, default="best_model.pth")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    return p.parse_args()

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_data_loaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected 'train' and 'val' subfolders under {data_dir}. "
                                f"Got {train_dir.exists()=}, {val_dir.exists()=}.")

    # Augment more on train, less on val/test
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True)
    class_names = train_dataset.classes
    return train_loader, val_loader, class_names

def build_model(num_classes=2, pretrained=True):
    # Compatible with both older and newer torchvision:
    try:
        # NOTE: torchvision >= 0.13 uses weights argument
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # fallback for older torchvision that expects pretrained boolean
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_names = get_data_loaders(args.data_dir, img_size=args.img_size,
                                                            batch_size=args.batch_size, num_workers=args.num_workers)
    print("Classes:", class_names)

    model = build_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print("Resuming from checkpoint:", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and ckpt["scheduler_state"]:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception as e:
                print("Warning: failed to load scheduler state:", e)
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch} with best_val_acc={best_val_acc:.4f}")

    print("Starting training for", args.epochs, "epochs")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            # save best model
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
                "best_val_acc": best_val_acc,
                "class_names": class_names
            }, filename=args.model_out)
        # also save last checkpoint
        save_checkpoint({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else {},
            "best_val_acc": best_val_acc,
            "class_names": class_names
        }, filename="last_checkpoint.pth")

        t1 = time.time()
        print(f"Epoch [{epoch+1}/{args.epochs}]  time: {t1-t0:.1f}s")
        print(f"  Train loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  Acc: {val_acc:.4f}  (best: {best_val_acc:.4f})")

    print("Training complete. Best val acc: {:.4f}".format(best_val_acc))
    print("Best model saved to:", args.model_out)

def predict_single_image(model_path, image_path, device=None, img_size=224):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    class_names = ckpt.get("class_names", ["class0", "class1"])
    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    from PIL import Image
    preprocess = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    return {"label": class_names[idx.item()], "confidence": float(conf.item())}

if __name__ == "__main__":
    main()