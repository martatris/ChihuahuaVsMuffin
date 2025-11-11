# Chihuahua vs Muffin Classifier (PyTorch)

This project trains a deep learning model (ResNet-18) to distinguish between images of **chihuahuas** and **muffins** using PyTorch.

## Features
- Transfer learning with pretrained ResNet-18
- Data augmentation for robustness
- Automatic mixed precision (AMP) support for faster training
- Checkpoint saving and resuming
- Learning rate scheduling
- Inference on single images

---

## Project Structure

```
project/
│
├── MuffsvsChi.py               # Main training script
├── data/
│   ├── train/
│   │   ├── chihuahua/
│   │   └── muffin/
│   └── val/
│       ├── chihuahua/
│       └── muffin/
├── best_model.pth         # Saved best model checkpoint
├── last_checkpoint.pth
└── ReloadApp.py
```

`best_model.pth`, `last_checkpoint.pth` and the data is to large to upload. You could find the dataset through kaggle, whereas the best model and the last checkpoint files are created after the code is run.

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision tqdm pillow
```

(Optional for CUDA users)
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Training

Run training from the command line:

```bash
python train.py --data-dir ./data --epochs 20 --batch-size 32 --use-amp
```

Available arguments:

| Argument | Description | Default |
|-----------|-------------|----------|
| `--data-dir` | Root folder containing `train/` and `val/` subfolders | `./data` |
| `--epochs` | Number of epochs to train | `15` |
| `--batch-size` | Batch size | `32` |
| `--lr` | Learning rate | `1e-3` |
| `--weight-decay` | Weight decay for regularization | `1e-4` |
| `--img-size` | Input image size | `224` |
| `--num-workers` | Data loader workers | `4` |
| `--model-out` | Path to save best model | `best_model.pth` |
| `--resume` | Resume training from checkpoint | `None` |
| `--use-amp` | Enable automatic mixed precision | `False` |
| `--optimizer` | Optimizer type: `adam` or `sgd` | `adam` |

---

## Example Command

```bash
python train.py --data-dir ./data --epochs 10 --batch-size 64 --optimizer sgd --use-amp
```

---

## Inference

You can predict a single image using the saved model:

```python
from train import predict_single_image

result = predict_single_image("best_model.pth", "test.jpg")
print(result)
```

Output example:
```
{'label': 'muffin', 'confidence': 0.982}
```

---

##  Checkpoints

During training, two checkpoints are saved:
- `best_model.pth`: best model by validation accuracy.
- `last_checkpoint.pth`: latest epoch model.

You can resume training from a checkpoint:

```bash
python train.py --resume last_checkpoint.pth
```


