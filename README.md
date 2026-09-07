# Malaria Diagnosis Support System (Simple Version)

A minimal implementation of the project described in the report:
a CNN (ResNet50 transfer learning) that classifies blood smear
images as **positive** or **negative** for malaria, served through
a small Flask web app.

Kept deliberately simple for a final year project — one Flask app,
plain HTML/JS frontend (no React/database/auth), one training script.

## Project structure

```
malaria-diagnosis/
├── app.py              # Flask web server + prediction endpoint
├── train_model.py      # Trains the ResNet50 model
├── requirements.txt
├── templates/
│   └── index.html      # Upload page
├── static/
│   ├── style.css
│   └── script.js
└── model/
    └── malaria_resnet50.h5   # (created after training)
```

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

The report used 8,500 in-house images. For a student project, use the
free public **NIH Malaria Cell Images** dataset (27,558 images),
which has the exact folder layout this code expects:

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Download and unzip it so you have:

```
dataset/
├── Parasitized/
└── Uninfected/
```

## 3. Train the model

```bash
python train_model.py --data_dir dataset/ --epochs 10
```

This freezes the first 40 layers of ResNet50, fine-tunes the rest,
and saves the trained model to `model/malaria_resnet50.h5`.
Training on CPU is slow — a GPU (even a free Colab one) is recommended.
If you don't have a GPU, try fewer epochs first to make sure everything runs.

## 4. Run the web app

```bash
python app.py
```

Open **http://localhost:5000**, upload a blood smear image, and
click Analyze.

If you haven't trained a model yet, the app still runs in **demo mode**
and returns a placeholder result, so you can build/test the UI first.

## Notes for your report

- Model: ResNet50 pre-trained on ImageNet, first 40 layers frozen,
  custom head = GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
- Input size: 224×224 RGB
- Loss: binary cross-entropy, Optimizer: Adam (lr=0.0001)
- This simplified version omits things mentioned in the full report
  (user auth, database, offline sync, multi-language UI, batch upload)
  to keep the codebase manageable for a student project. You can
  mention these as "future work" / recommendations, which the report
  already does in Chapter 5.
