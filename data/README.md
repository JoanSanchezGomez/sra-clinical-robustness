# Datasets for SRA Clinical Robustness

This project uses two main datasets: **EDC** and **ODIR-5K**. Below are the instructions to download and organize them.

## 📁 Directory Structure

After downloading, your `data/` folder should look like this:
data/
│
├── eye-diseases-classification/ # EDC dataset
│ ├── normal/
│ ├── diabetic_retinopathy/
│ ├── glaucoma/
│ └── cataract/
│
└── ODIR-5K/ # ODIR-5K dataset
├── Training Images/
│ ├── 0_left.jpg
│ ├── 0_right.jpg
│ └── ...
└── data.xlsx


---

## 👁️ EDC (Eye Diseases Classification)

- **Source**: [Kaggle - Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Samples**: ~4,000 images
- **Classes**: 4 (cataract, diabetic_retinopathy, glaucoma, normal)
- **Type**: Single-label classification
- **Balanced**: Yes

### Download Instructions:
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
2. Download the dataset (you may need a Kaggle account)
3. Extract the contents to `data/eye-diseases-classification/`

---

## 🏥 ODIR-5K (Ocular Disease Intelligent Recognition)

- **Source**: [Kaggle - ODIR-5K Classification](https://www.kaggle.com/datasets/tanjemahamed/odir5k-classification)
- **Samples**: ~8,000 images (left and right eyes of 4,000 patients)
- **Classes**: 8 ocular disease categories
- **Type**: Multi-label classification
- **Balanced**: No (imbalanced: Normal has 2,873 samples, Hypertension has 128)

### Download Instructions:
1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/tanjemahamed/odir5k-classification)
2. Download the dataset
3. Extract the contents to `data/ODIR-5K/`
4. Ensure the folder contains:
   - `Training Images/` folder with all images
   - `data.xlsx` file with labels

---

## 📊 Additional Datasets (Optional)

The codebase can be extended to support these datasets:

| Dataset | Link | Description |
|---------|------|-------------|
| Messidor-2 | [Kaggle](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess) | Diabetic retinopathy grading |
| MuReD | [IEEE Dataport](https://ieee-dataport.org/documents/multi-label-retinal-disease-mured-dataset) | Multi-label retinal diseases |
| RFMiD 2.0 | [IEEE Dataport](https://ieee-dataport.org/documents/retinal-fundus-multi-disease-image-dataset-rfmid-20) | 51 retinal disease classes |

---

## ⚠️ Important Notes

- **Do not upload** the datasets to GitHub (they are excluded via `.gitignore`)
- The code expects the exact folder structure shown above
- Make sure you have enough disk space (datasets can be several GB)
- Some datasets require registration or acceptance of terms of use
