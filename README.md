# Van Gogh Painting Identification

A lightweight, CPU-friendly deep learning workflow to classify whether a painting is by Vincent van Gogh using the **VGDB-2016** dataset. This project features patch-based modeling, modular training pipelines, and support for ResNet, ViT, EfficientNet, and more.

---

## Project Highlights

* **Notebook-based pipeline** from EDA to final evaluation.
* **Patch-based classification** for enhanced attribution precision.
* **Multiple model backbones** (ResNet, VGG, EfficientNet, ViT).
* **Inference-ready scripts** for both full image and patch-level voting.
* **Future-ready**: extension plans include GAN and diffusion model upgrades.

## Repository Layout

```text
.
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_model.ipynb
│   └── 04_predict.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── predict.py
│   ├── preprocessing_patches.py
│   └── trainer.py
```

## Libraries Used

* **torch**, **torchvision** – for deep learning and model architectures
* **pandas**, **numpy** – for data handling and manipulation
* **matplotlib** – for visualization
* **Pillow (PIL)** – for image loading and transformation
* **scikit-learn** – for metrics like F1-score
* **jupyter** – for running and organizing notebooks

## Quick Start (local, CPU‑only)

```bash
# 1. Clone the repository
https://github.com/your‑fork/van-gogh-identification.git
cd van-gogh-identification

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Launch notebooks
jupyter lab notebooks/01_EDA.ipynb
```

> **Note:** All notebooks are modular and can be run independently. Patch-based model training is optional.

## Patch-Based Prediction

Patchified inference splits a painting into 224×224 tiles and votes across their predictions. This makes it robust to localized noise and brushstroke patterns.

## Citation

If you use this project or dataset, please cite the original ICIP 2016 paper:

```
@InProceedings{folego2016vangogh,
  title={From Impressionism to Expressionism: Automatically Identifying Van Gogh's Paintings},
  author={Folego, Guilherme and Gomes, Otavio and Rocha, Anderson},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  pages={141--145},
  year={2016},
  doi={10.1109/icip.2016.7532335}
}
```
