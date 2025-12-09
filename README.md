Multimodel Forgery Detection — README

A clear, reproducible README for a multimodel forgery detection system trained on multiple face-forgery datasets (FF++, Real vs Fake, Celeb-DF).
This project trains and compares three backbone architectures — ResNet50, EfficientNet-B4, and Xception — and provides a Streamlit app to run inference with trained checkpoints.

Table of contents

Project overview

Repository layout (recommended)

Requirements & setup

Preparing the datasets

Training (how to run the training code)

Model evaluation & expected outputs

Running the Streamlit inference app

Tips, best practices & troubleshooting

Citation & license

1. Project overview

Purpose: train and evaluate multiple CNN backbones for face forgery detection, compare results, and serve an interactive web app to test images/videos.

Datasets used: FaceForensics++ (FF++), Real vs Fake dataset, Celeb-DF.

Backbones: ResNet50, EfficientNet-B4, Xception.

Main components:

train.py — training script (single script that accepts --model argument to choose backbone).

eval.py — evaluation script (compute accuracy, AUC, confusion matrix).

inference.py — inference utilities used by Streamlit app.

app.py — Streamlit app to upload image/video and run model(s).

utils/ — data loaders, augmentation, metrics, checkpoint handlers.

2. Recommended repository layout
forgery-detector/
├─ data/
│  ├─ ffpp/
│  ├─ real_vs_fake/
│  └─ celeb_df/
├─ checkpoints/
│  ├─ resnet50/
│  ├─ efficientnet_b4/
│  └─ xception/
├─ src/
│  ├─ train.py
│  ├─ eval.py
│  ├─ inference.py
│  ├─ app.py
│  ├─ models/
│  │  ├─ resnet50.py
│  │  ├─ efficientnet_b4.py
│  │  └─ xception.py
│  └─ utils/
│     ├─ dataloader.py
│     ├─ augmentations.py
│     └─ metrics.py
├─ requirements.txt
└─ README.md

3. Requirements & setup
Python environment

Python 3.8+ recommended.

Create and activate virtualenv:

python -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate       # Windows PowerShell

Install dependencies

Create requirements.txt (example):

torch>=1.12
torchvision
efficientnet-pytorch
tqdm
scikit-learn
opencv-python
pillow
pandas
numpy
matplotlib
streamlit
albumentations


Install:

pip install -r requirements.txt

GPU / CUDA

Training is GPU-intensive. Use CUDA-enabled PyTorch build.

For mixed-precision training, install torch.cuda.amp (part of modern PyTorch).

4. Preparing the datasets
Organize folders like:
data/
  ffpp/
    train/
      real/
      fake/
    val/
  real_vs_fake/
    train/
    val/
  celeb_df/
    train/
    val/

Recommendations:

Preprocess: extract faces (alignment optional) using MTCNN/face detector. Save face crops as images for faster training.

Use consistent resolution across datasets (e.g., 224×224 for ResNet50/EfficientNet; Xception typically 299×299 — but you can resize to 224 for uniformity if needed).

Balance classes during training or use weighted sampler if classes are imbalanced.

5. Training — how to run the training code

Below are example commands and a line-by-line explanation of the important CLI flags.

Example: train ResNet50
python src/train.py \
  --model resnet50 \
  --data-root data/ \
  --dataset ffpp,real_vs_fake,celeb_df \
  --train-split train \
  --val-split val \
  --epochs 30 \
  --batch-size 32 \
  --img-size 224 \
  --lr 1e-4 \
  --pretrained \
  --save-dir checkpoints/resnet50 \
  --device cuda:0

Example: train EfficientNet-B4
python src/train.py \
  --model efficientnet_b4 \
  --data-root data/ \
  --dataset ffpp,real_vs_fake,celeb_df \
  --epochs 30 \
  --batch-size 16 \
  --img-size 380 \
  --lr 1e-4 \
  --pretrained \
  --save-dir checkpoints/efficientnet_b4

Example: train Xception
python src/train.py \
  --model xception \
  --img-size 299 \
  --batch-size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --pretrained \
  --save-dir checkpoints/xception

CLI flag explanations (line-by-line)

python src/train.py — launches the training script.

--model resnet50 — selects backbone: resnet50, efficientnet_b4, or xception.

--data-root data/ — root folder where dataset subfolders live.

--dataset ffpp,real_vs_fake,celeb_df — comma-separated datasets to include during training (the script should combine them).

--train-split train / --val-split val — subfolders inside each dataset (flexible).

--epochs 30 — number of epochs to train.

--batch-size 32 — batch size per GPU (reduce if OOM).

--img-size 224 — resize images to this size before feeding network.

--lr 1e-4 — initial learning rate.

--pretrained — initialize model with ImageNet weights (recommended).

--save-dir checkpoints/resnet50 — where model checkpoints & logs are stored.

--device cuda:0 — CUDA device; use cpu if no GPU.

What the training script should do (expected behavior)

Load dataset(s), apply augmentations.

Build model according to --model.

Use an optimizer (Adam/SGD) and a scheduler (CosineLR or ReduceLROnPlateau).

Save best model by validation AUC/accuracy.

Log losses, metrics to console (and optionally to a CSV file).

6. Model evaluation & expected outputs
After training:

Checkpoint files: checkpoints/<model>/best.pth and last.pth.

Training logs: checkpoints/<model>/training_log.csv (epoch, train_loss, val_loss, val_acc, val_auc).

Example evaluation output (console):

Epoch 30/30
Train Loss: 0.065, Train Acc: 0.97
Val Loss: 0.128, Val Acc: 0.94, Val AUC: 0.987
Saved best checkpoint: checkpoints/resnet50/best.pth

Metrics to compute:

Accuracy (binary).

ROC AUC (recommended for imbalance).

Confusion matrix (TP, TN, FP, FN).

Precision / Recall / F1.

Per-dataset split performance (report FF++, Celeb-DF, RealVsFake separately to assess generalization).

7. Running the Streamlit app (serve models for inference)
Run Streamlit
streamlit run src/app.py --server.port 8501

What app.py should do (expected behavior)

Provide UI to upload image(s) or video.

Let user choose the model checkpoint to use (ResNet50 / EfficientNet-B4 / Xception).

Load the selected checkpoint and run inference pipeline:

Detect & crop face(s) (or assume already-cropped face).

Resize appropriately (depending on model).

Predict probability of fake.

Display:

Predicted label (Real / Fake).

Confidence score (e.g., probability 0–1).

Optionally overlay heatmap (Grad-CAM) for explanation.

Example UI workflow

Upload image (or video).

Select model & checkpoint.

Click Run.

See results: prediction, confidence, sample predictions for multiple faces.

8. Tips, best practices & troubleshooting
Training tips

Use --pretrained and fine-tune; freeze early layers for first few epochs if unstable.

Use torch.cuda.amp (mixed precision) for faster training and lower memory usage.

Use data augmentation (random crop, flip, color jitter) but avoid unrealistic transforms.

Keep validation dataset strictly separate from training sources to gauge generalization.

If OOM, reduce --batch-size or --img-size.

Save model checkpoints at every epoch and retain the best by validation AUC.

Hyperparameter suggestions

lr = 1e-4 (Adam), batch = 16–32 (depending on GPU), epochs = 25–50.

For EfficientNet-B4, use larger img-size (e.g., 380) and smaller batch if GPU memory limited.

Evaluation & domain-shift

Test cross-dataset: train on FF++ and RealVsFake, test on Celeb-DF (or vice versa) to measure robustness.

Report per-forgery-type performance (if FF++ labels include DeepFakes, Face2Face, etc.)

Common troubleshooting

Low validation AUC: reduce learning rate, check dataset leakage (same identities in train & val), increase data augmentation.

Very fast overfitting: use dropout, lower learning rate, or add more data.

Slow training: use mixed precision and increase workers in DataLoader.

Streamlit freezing on load: ensure app.py lazy-loads the model upon first request rather than at import time.

9. Example minimal train.py patterns (pseudocode)

This is a high-level pseudocode — ensure you adapt to your actual training code.

# src/train.py (pseudocode)
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['resnet50','efficientnet_b4','xception'])
...
args = parser.parse_args()

# 1. Build dataset and dataloaders
train_ds = Dataset(args.data_root, split='train', transforms=train_transforms)
val_ds = Dataset(args.data_root, split='val', transforms=val_transforms)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

# 2. Build model
model = build_model(args.model, pretrained=args.pretrained, num_classes=1)

# 3. Optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# 4. Training loop with amp
for epoch in range(args.epochs):
    train_one_epoch(...)
    val_metrics = validate(...)
    scheduler.step(val_metrics['loss'])
    save_checkpoint_if_best(...)

10. Reproducibility checklist

Save exact requirements.txt and CUDA/PyTorch version.

Save training config (hyperparameters) in checkpoints/<model>/config.json.

Fix seeds for reproducibility (torch.manual_seed, np.random.seed, random.seed).

Log code git commit hash with results.

11. Suggested experiments & extensions

Ensembling: average probabilities of all three backbones to boost accuracy.

Domain adaptation: adversarial domain adaptation or fine-tune on a small subset of target dataset for better generalization.

Temporal models: for videos, use frame sequences + temporal networks (LSTM / 3D-CNN).

Explainability: integrate Grad-CAM visualizations in Streamlit.

12. License & citation

State your project license (e.g., MIT).

When publishing results, cite FF++ and Celeb-DF datasets as required by their licenses/usage guidelines.

13. Quick-start checklist (short)

Set up virtualenv and pip install -r requirements.txt.

Prepare datasets and place under data/ as described.

Train models:

python src/train.py --model resnet50 --data-root data/ ...

Repeat for efficientnet_b4 and xception.

Run Streamlit:

streamlit run src/app.py

Open http://localhost:8501 and test.
