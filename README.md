# Stanford Dogs CNN — What Actually Improves Performance?

A structured empirical study on the Stanford Dogs dataset (120 breeds, 20,580 images)
exploring what actually moves the needle in CNN performance from a 7% baseline to 84%
through controlled experimentation.

---

## Final Results

| Approach | Val Accuracy |
|---|---|
| Scratch CNN baseline | 7.40% |
| Optuna tuned scratch CNN | 14.53% |
| MobileNetV2 FE 128×128 | 75.44% |
| EfficientNetB0 FE 128×128 | 70.21% |
| NASNetMobile FE 128×128 | 56.63% |
| ResNet50 FE 128×128 | 56.56% |
| MobileNetV2 FT 128×128 | 72.55% |
| MobileNetV2 FE 224×224 | 84.06% ← best |
| MobileNetV2 FT 224×224 | 81.00% |

---

## Project Structure

| Notebook | Topic | Key Result |
|---|---|---|
| 1_data_and_baseline | Data exploration + baseline CNN | 7% val accuracy — reference point |
| 2_input_size_experiment | Input size: 64 vs 128 vs 224 | 128×128 chosen for experiments 3–7 |
| 3_architecture_experiments | BatchNorm, GAP, depth, filters, padding | Architecture matters more than tuning |
| 4_regularization_experiments | Dropout, L2, augmentation | Augmentation = biggest overfitting fix |
| 5_manual_hyperparameter_tuning | LR, batch size, optimizer | LR most impactful single parameter |
| 6_optuna_tuning | Automated Bayesian search, 20 trials | 14.53% — matched manual tuning |
| 7_transfer_learning_feature_extraction | 4 pretrained models compared | MobileNetV2 won at 75.44% |
| 8_fine_tuning | Fine tuning at 128×128 | Fine tuning hurt — 72.55% vs 75.68% |
| 8b_fine_tuning_improved | Fine tuning at 224×224 | 84.06% — best result in project |
| 9_final_comparison | Every experiment side by side | Full analysis and conclusions |

---

## Key Findings

**What moved the needle most**
Transfer learning and input size. A frozen MobileNetV2 jumped straight to 75%
where the best scratch CNN with Optuna tuning only reached 14.5%. Going from
128×128 to 224×224 added another 8% on top.

**What was overhyped**
Optuna. It's a great tool but it can only search within the space you give it.
When the architecture is the bottleneck, automated search can't fix it.
Fine tuning also consistently underperformed feature extraction because the base
model was already overfitting before layers were unfrozen.

**What I'd do differently**
Start at 224×224 from the beginning, add data augmentation before fine tuning,
and skip manual tuning experiments on scratch CNNs entirely for a dataset this hard.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Optuna
- Pandas
- NumPy
- Matplotlib
- Google Colab
- Lightning AI

---

## Dataset

Stanford Dogs Dataset — available on
[Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

120 breeds, 20,580 images, ~170 images per class.

```python
import kagglehub
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
```

---

## Setup

```bash
pip install tensorflow optuna kagglehub matplotlib pandas numpy
```

Run notebooks in order from 1 through 9. Each notebook reloads data fresh
at the top — no state carried between notebooks.
