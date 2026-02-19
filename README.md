# ASR: Systemic Relationship Analysis for Clinical Robustness Auditing in Ophthalmic AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Description

This repository contains the official implementation of the **ASR (Systemic Relationship Analysis)** method and the **F1-RS (F1-Robustness Score)** metric for auditing the clinical robustness of artificial intelligence models in ophthalmic diagnosis.

The ASR method enables:
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
git clone https://github.com/JoanSanchezGomez/asr-clinical-robustness.git
cd asr-clinical-robustness

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
