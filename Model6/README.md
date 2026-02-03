
---

## Forest Cover Type Classification

### Dataset

* **Source:** Kaggle — *Forest Cover Type Dataset*
* **Author:** UC Irvine Machine Learning Repository
* **Link:** [https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)

The dataset contains cartographic variables (such as elevation, soil type, and proximity to hydrology) for forested areas, along with a target label (**Cover_Type**) indicating the forest cover category.

---

### Objective

To build a **multiclass classification model** that predicts the forest cover type based on geographic and environmental features, using robust preprocessing, scaling, and evaluation.

---

### Exploratory Data Analysis (EDA)

* **Correlation analysis** to inspect linear relationships and feature dependencies.
* **Class distribution plots** to visualise sample balance or imbalance across cover types.
* **Feature distribution visualisations** (histograms/boxplots) to explore how variables differ between cover types.

---

### Preprocessing Pipeline

Implemented using **scikit‑learn Pipelines** to ensure leakage‑safe transformations.

**Numerical features**

* Standardisation (`StandardScaler`) to normalise continuous variables.

**Categorical variables (if present)**

* One‑hot encoding with graceful handling of unknown categories.

Class imbalance is preserved via **stratified train–test splitting** and can be further addressed if needed using sampling techniques or class weighting.

---

### Models

* **Algorithms:**

  * Multinomial Logistic Regression
  * One‑vs‑Rest Logistic Regression
  * One‑vs‑One Logistic Regression
* **Max iterations:** 500
* **Scaling:** StandardScaler
* All models are integrated into pipelines to prevent leakage and ensure reproducibility.

---

### Training & Evaluation

* **Train–test split:** 80/20
* **Stratification:** Preserves class distribution
* **Metrics:**

  * Accuracy
  * Precision
  * Recall
  * F1‑score
  * Confusion matrix
  * ROC–AUC (per class, weighted for multiclass)

Metrics are chosen to balance overall correctness with sensitivity across all forest cover categories.

---

### Key Characteristics

* Fully pipeline‑driven (scaling and modelling combined)
* Leakage‑safe and reproducible
* Supports multiclass evaluation and ROC–AUC visualisation
* Structured for experimentation and hyperparameter tuning

---

### Use Case

This model serves as a **baseline forest cover classifier**, appropriate for:

* Educational machine learning projects
* Comparative benchmarking of classification algorithms
* Geospatial and ecological feature analysis
* Multiclass performance evaluation and optimisation

---
