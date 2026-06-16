# TML26 Task 3 — Adversarial Robustness

**Best leaderboard score: 0.662440**

## Overview

Train a ResNet50 image classifier on 50,000 images (3×32×32, 9 classes) using PGD adversarial training (Madry et al., 2018). The final score is the average of clean accuracy and robust accuracy on adversarially perturbed test inputs.

## Architecture

- **Model:** ResNet50 (`torchvision.models.resnet50`) with the final `fc` layer replaced to output 9 classes
- **Training:** PGD adversarial training from epoch 1 — each batch trains on both clean (augmented) and adversarial examples concatenated together
- **PGD parameters:** ε = 0.04, α = 2/255, 10 iterations per step
- **Augmentation:** RandomCrop (padding=4), RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(90°)
- **Optimizer:** SGD with momentum=0.9, weight_decay=5e-4, Nesterov; LR drops at epochs 60 and 90 (0.1 → 0.01 → 0.001)
- **Loss:** CrossEntropyLoss with label smoothing=0.1
- **Epochs:** 120; best checkpoint saved by unified score (0.5 × clean + 0.5 × robust)

## Prerequisites

```
pip install torch torchvision tqdm requests numpy
```

You also need:
- `train.npz` — the training dataset (download from HuggingFace as linked in the assignment)
- Your team API key for submission

## Reproducing the Best Result

### Step 1 — Train with PGD adversarial training

Open [pgd_attack.py](pgd_attack.py) and update the data path on line 21 to point to your local `train.npz`:

```python
data = np.load("/path/to/train.npz")
```

Then run:

```bash
python pgd_attack.py
```

This trains for 120 epochs and saves the best checkpoint as `model_pgd.pt`. 

### Step 2 — Submit

Open [submission.py](submission.py) and set your API key:

```python
API_KEY = "YOUR_API_KEY_HERE"   # replace with your actual API key
```

Verify the model path and architecture are correct (defaults match the training script):

```python
MODEL_PATH = "model_pgd.pt"
MODEL_NAME = "resnet50"
SUBMIT = True
```

Then run:

```bash
python submission.py
```
