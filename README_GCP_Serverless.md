# ☁️ DSA5208 Weather Prediction — Google Cloud Serverless ML Pipeline

This repository implements a **serverless PySpark machine learning pipeline** on **Google Cloud Dataproc Batches** for large-scale weather prediction.  
All stages — from data cleansing to model training (Linear, Lasso, Random Forest, XGBoost) — are executed fully serverless using **Dataproc Batches** and **Google Cloud Storage (GCS)**.  
Results and evaluation reports are generated automatically in Markdown and PDF.

---

## 🌐 Cloud Architecture

Below is the high-level flow of the serverless ML pipeline:

```
          ┌───────────────────────────────┐
          │        Local Machine          │
          │  (Development & Submission)   │
          └──────────────┬────────────────┘
                         │ gcloud CLI
                         ▼
          ┌───────────────────────────────┐
          │   Google Cloud Storage (GCS)  │
          │   ├── src/ (PySpark scripts)  │
          │   ├── data/ (Parquet files)   │
          │   └── results_/ (Outputs)     │
          └──────────────┬────────────────┘
                         │
                         ▼
          ┌───────────────────────────────┐
          │  Dataproc Serverless (Batch)  │
          │  ├── Auto-scales compute      │
          │  ├── Executes PySpark jobs    │
          │  └── Writes outputs to GCS    │
          └──────────────┬────────────────┘
                         │
                         ▼
          ┌───────────────────────────────┐
          │     Results on GCS Bucket     │
          │  metrics_csv / plots / logs   │
          │  feature_importance_csv       │
          └──────────────┬────────────────┘
                         │
                         ▼
          ┌───────────────────────────────┐
          │ Local Report Generator        │
          │ (model_report_to_pdf.py)      │
          │ → Markdown + PDF summary      │
          └───────────────────────────────┘
```

---

## 🧭 1. Prerequisites

Install and authenticate Google Cloud SDK:

```bash
brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project <your-project-id>
gcloud config set dataproc/region asia-southeast1
```

---

## 🪣 2. Create the Google Cloud Storage Bucket

```bash
gsutil mb -l asia-southeast1 gs://dsa5208-mllib-weather-predict
gsutil ls -p <your-project-id>
```
Upload scripts and data:
```bash
gsutil -m cp -r src/ gs://dsa5208-mllib-weather-predict/src/
gsutil -m cp -r data/ gs://dsa5208-mllib-weather-predict/data/
```

---

## ⚙️ 3. Submit Serverless Jobs

Each stage is executed via Dataproc Batches.

### Data Cleanse
```bash
gcloud dataproc batches submit pyspark gs://dsa5208-mllib-weather-predict/src/data_cleanse.py   --region=asia-southeast1 --version=2.3   --deps-bucket=gs://dsa5208-mllib-weather-predict   --properties="spark.sql.shuffle.partitions=200,spark.default.parallelism=200"   --   --input gs://dsa5208-mllib-weather-predict/data/compacted/2024.parquet
```

### Linear Regression
```bash
gcloud dataproc batches submit pyspark gs://dsa5208-mllib-weather-predict/src/base_linearReg.py   --region=asia-southeast1 --version=2.3   --deps-bucket=gs://dsa5208-mllib-weather-predict   --properties="spark.sql.shuffle.partitions=200,spark.default.parallelism=200"   --   --input gs://dsa5208-mllib-weather-predict/data/train_2024.parquet   --input gs://dsa5208-mllib-weather-predict/data/test_2024.parquet
```

### Ridge / Lasso Regression
```bash
gcloud dataproc batches submit pyspark gs://dsa5208-mllib-weather-predict/src/Ridge_Lasso.py   --region=asia-southeast1 --version=2.3   --deps-bucket=gs://dsa5208-mllib-weather-predict   --properties="spark.sql.shuffle.partitions=200,spark.default.parallelism=200"   --   --input gs://dsa5208-mllib-weather-predict/data/train_2024.parquet   --input gs://dsa5208-mllib-weather-predict/data/test_2024.parquet
```

### Random Forest Regression
```bash
gcloud dataproc batches submit pyspark gs://dsa5208-mllib-weather-predict/src/rf_train_eval_all_cloud.py   --region=asia-southeast1 --version=2.3   --batch="rf-$(date +%Y%m%d-%H%M%S)"   --deps-bucket=gs://dsa5208-mllib-weather-predict   --ttl=1d   --properties="spark.driver.memory=20g,spark.driver.maxResultSize=0"   --   --train gs://dsa5208-mllib-weather-predict/data/train_2024.parquet   --test  gs://dsa5208-mllib-weather-predict/data/test_2024.parquet   --numTrees 120 --maxDepth 12 --subsample 0.8 --partitions 300 --driver_mem 20g --make_figs
```

### XGBoost Regression
```bash
gcloud dataproc batches submit pyspark gs://dsa5208-mllib-weather-predict/src/XGBoost.py   --region=asia-southeast1 --version=2.3   --deps-bucket=gs://dsa5208-mllib-weather-predict   --ttl=1d   --properties="spark.executorEnv.USE_GCS=1,spark.executorEnv.GCS_BUCKET=dsa5208-mllib-weather-predict"   --   --train gs://dsa5208-mllib-weather-predict/data/train_2024.parquet   --test  gs://dsa5208-mllib-weather-predict/data/test_2024.parquet   --ycol TMP_C   --outdir gs://dsa5208-mllib-weather-predict/results_xgb_full   --n_iter 20 --cv 3 --threads 8 --max_bins 256 --max_cat 128 --no_plots
```

---

## 🧾 4. Local Report Generation

Once model outputs are stored in GCS or local results, generate Markdown + PDF reports:

```bash
python src/model_report_to_pdf.py   --results_dir results_cloud/rf_results   --model_name "Random Forest Regressor (Weather TMP_C)"   --target_col TMP_C

python src/model_report_to_pdf.py   --results_dir results_cloud/xgb_results   --model_name "XGBoost Regressor (Weather TMP_C)"   --target_col TMP_C
```

---

## 📦 5. Requirements

### requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.8.0
markdown2>=2.4.0
pypandoc>=1.11
tabulate>=0.9.0
openpyxl>=3.1.2
reportlab>=4.0.0
argparse>=1.4.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

© 2025 Koushik — MSc Data Science & Sustainability (DSA5208 Project)
