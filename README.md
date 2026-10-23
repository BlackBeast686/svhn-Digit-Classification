# SVHN Digit Classification

A computer vision assignment comparing three models for **digit classification on the SVHN (Street View House Numbers) dataset**. Each model is trained on the same 32×32 colour digit images and evaluated on the same test split, moving from a simple linear baseline up to a convolutional network to show how much representation learning matters for image data.

> **Note on scope:** This is a research/coursework notebook for comparing model families, not a production system.

## Task & data

- **Inputs:** 32×32×3 RGB images of house-number digits
- **Outputs:** one of 10 classes (digits 0–9; the dataset's label `10` is remapped to `0`)
- **Dataset:** SVHN (cropped format), ~73k training and ~26k test images

The dataset is **not** included in this repository. The notebook downloads it directly from Stanford:

- `train_32x32.mat` and `test_32x32.mat` from http://ufldl.stanford.edu/housenumbers/

For the two baseline models, the images are flattened to 3072-dimensional vectors; the CNN uses the full image tensor.

## Models

| Model | Type | Input |
|-------|------|-------|
| Logistic Regression | Linear baseline (scikit-learn) | Flattened pixels |
| Feedforward Neural Network (MLP) | One hidden layer of 64 units (scikit-learn) | Flattened pixels |
| CNN | 3 conv + 2 pooling layers, dense head (TensorFlow/Keras) | Full 32×32×3 image |

## Results

Test-set accuracy as reported in the notebook:

| Model | Accuracy |
|-------|----------|
| CNN | 0.904 |
| Feedforward NN (MLP) | 0.420 |
| Logistic Regression | 0.258 |

> The baselines train on flattened pixels (and are capped at low iteration counts), so their low scores are expected — the point of the comparison is the large jump the CNN delivers by exploiting spatial structure. Figures are from single runs.

## Requirements

Developed in Google Colab. Main dependencies:

- `tensorflow`
- `scikit-learn`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`

Install everything with:

```bash
pip install -r requirements.txt
```

## Getting started

1. Open the notebook in Google Colab or Jupyter.
2. Run the cells top to bottom — the first cell downloads the SVHN `.mat` files automatically (if `wget` is unavailable locally, download the two files from the link above and place them next to the notebook).
3. The notebook trains all three models and plots their confusion matrices and the CNN's training curve. A GPU speeds up the CNN but is not required.

## Repository structure

```
.
├── CV_Assignment.ipynb
└── README.md
```
