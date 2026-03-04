# SRA: Systemic Relationship Analysis for Clinical Robustness Auditing in Ophthalmic AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Description

This repository contains the official implementation of the **SRA (Systemic Relationship Analysis)** method and the **F1-RS (F1-Robustness Score)** metric for auditing the clinical robustness of artificial intelligence models in ophthalmic diagnosis.

The SRA method enables:
- Automatic identification of the most critical diagnostic contrasts
- Quantification of robustness degradation through ΔF1-RS
- Detection of "robustness illusion" in apparently accurate models

## 🚀 Supported Architectures

- **EfficientNet-B0** (lightweight CNN)
- **ConvNeXt-Base** (modern CNN)
- **Swin Transformer Tiny** (hierarchical Transformer)
- And all models available in `timm`

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/JoanSanchezGomez/sra-clinical-robustness.git
cd sra-clinical-robustness

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## 📥 Data Download

This project requires two datasets. Follow the instructions below to download and organize them.

### Required Datasets

| Dataset | Source | Size | Type |
|---------|--------|------|------|
| **EDC** | [Kaggle - Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) | ~4k images | Single-label |
| **ODIR-5K** | [Kaggle - ODIR-5K Classification](https://www.kaggle.com/datasets/tanjemahamed/odir5k-classification) | ~8k images | Multi-label |

### Quick Setup

```bash
# Create data directory
mkdir -p data

# For detailed instructions, see:
# - data/README.md
# - scripts/download_datasets.sh
