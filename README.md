# Restaurant Aggregate Rating Predictor

An end-to-end machine learning project designed to predict the aggregate rating of restaurants worldwide based on contextual features like cost, cuisine, online delivery options, location, and user engagement (votes). 

This project implements a structured data preprocessing pipeline featuring missing value imputation, categorical encoding, feature scaling, and ensemble modeling via the **Random Forest Regressor** to achieve an R² score of **95.54%**.

---

## 📂 Dataset Overview
The model utilizes a global restaurant dataset containing the following variables:

* **Target Variable:** `Aggregate rating` (Continuous scale from 0 to 5)
* **Numerical Features:** `Votes`, `Average Cost for two`, `Price range`, `Country Code`
* **Categorical Features:** `Cuisines`, `City`, `Has Table booking`, `Has Online delivery`

---

## 🛠️ Architecture & Pipeline
To eliminate data leakage and ensure robust generalization, the workflow utilizes a Scikit-Learn `Pipeline` combined with a `ColumnTransformer`:

1. **Missing Value Imputation:** * Numerical columns are imputed using the `median` strategy to minimize outlier influence.
   * Categorical columns are imputed using the `most_frequent` (mode) strategy.
2. **Categorical Encoding:** High-cardinality columns (`Cuisines`, `City`) are processed using `OneHotEncoder` with `handle_unknown='ignore'` to robustly capture new variations during inference.
3. **Feature Scaling:** Continuous numeric columns are scaled via `StandardScaler`.
4. **Data Splitting:** The data is split into an 80% training set and a 20% testing set.

---

## 📊 Model Performance Evaluation

Two regression algorithms were evaluated against the testing matrix:

| Regression Algorithm | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R-squared (R² Score) |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | *Degraded due to multi-collinearity* | — | — |
| **Random Forest Regressor** | **0.1014** | **0.3184** | **0.9554** |

### Insights:
* **Linear Regression** suffers significantly due to extreme multi-collinearity and sparsity introduced by one-hot encoding thousands of unique cuisine combinations. 
* **Random Forest Regressor** successfully handles these complex, non-linear interactions, mapping the underlying dependencies with high precision.

---

## 📈 Feature Influence Analysis

Extracting the internal feature importance weights from the trained Random Forest model reveals the primary driving metrics behind restaurant ratings:

| Rank | Feature | Importance Weight | Description |
| :--- | :--- | :--- | :--- |
| 1 | **Votes** | **94.94%** | Total customer review volume (Dominant Driver) |
| 2 | **Country Code** | **0.85%** | Geographic macro-region |
| 3 | **Average Cost for two** | **0.68%** | Pricing tier profile metrics |
| 4 | **City (New Delhi)** | **0.13%** | Local structural density indicator |
| 5 | **City (Noida)** | **0.11%** | Regional metropolitan indicator |

### Analytic Deduction:
User engagement (`Votes`) captures nearly the entirety of the model's variance. This indicates a structural pattern where highly rated restaurants attract significantly larger numbers of voting users, or unrated/new restaurants with zero votes default to a baseline score.

---

## 🚀 Getting Started

### 📦 Prerequisites
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn

# Comprehensive Guide: Running the Restaurant Predictor in Google Colab

This guide provides a step-by-step walkthrough to upload your dataset, configure the environment, and execute the machine learning pipeline using Google Colab.

---

### Step 1: Open Google Colab and Create a Notebook
1. Navigate to [colab.research.google.com](https://colab.research.google.com/) in your web browser.
2. Sign in with your Google account.
3. In the pop-up modal, click **New Notebook** at the bottom right. A fresh workspace will initialize.

---

### Step 2: Upload Your Dataset
Google Colab operates on dynamic cloud instances, meaning files must be uploaded to your active session before execution:
1. Locate the left-hand sidebar menu and click the **Folder icon** ($\square$) to open the Files panel.
2. Click the **Upload to session storage** button (the icon showing a page with an upward arrow).
3. Select your local `Dataset .csv` file and click open.
4. Click **OK** on the runtime reminder popup notice.

*Note: Session storage resets when your browser tab is closed for an extended period. You will simply need to re-upload the CSV using this folder panel when returning to the notebook.*

---

### Step 3: Code Implementation and Execution
Colab uses interactive code segments called **Cells**. Click the **`+ Code`** button in the upper left corner to add cells as needed.

#### Cell 1: Environment Optimization
Paste the following installation command into your first cell and press **Shift + Enter** (or click the circular **Play** button on the left of the cell) to run it:

```python
!pip install --upgrade pandas numpy scikit-learn
