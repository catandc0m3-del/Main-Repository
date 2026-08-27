# CI2604P — Machine Learning Project Brief
## Smoker Status Prediction Challenge

Welcome to your **CI2604P Machine Learning Project**. In this assignment, you will apply the end-to-end machine learning workflow to build, evaluate, and refine a predictive model using physiological bio-signal data.

---

## Project Overview

Identifying health risk factors using biological signals is a core application of data analytics and artificial intelligence in medicine. Your objective in this project is to develop a binary classification model that predicts an individual's smoking status (**`1`** for Smoker, **`0`** for Non-Smoker) based on physical attributes, cardiovascular metrics, and laboratory test results.

> [!NOTE]
> ### Kaggle Competition Link
> Access the competition, download datasets, and track the live leaderboard here:  
> **[CI2604P Kaggle Competition Entry Link](https://www.kaggle.com/t/2f773e1a9df748cc8fa3b2cc71cc714b)**

---

## Dataset & Context

* **Synthetic Data Generation:** The competition dataset (`train.csv` and `test.csv`) was synthetically generated using a deep learning model trained on the public *Smoker Status Prediction using Bio-Signals* dataset. Feature distributions are close to real-world bio-signals, but subtle nuances exist.
* **External Data Usage:** You are **explicitly permitted** to use the original public *Smoker Status Prediction using Bio-Signals* dataset. You may use it for exploratory data analysis (EDA), to compare feature distributions, or to combine it with your training set to evaluate whether it improves model generalization.

### Feature Dictionary (24 Variables)

The training dataset comprises the following 24 variables (23 input features + 1 target variable):

| Variable | Description |
| :--- | :--- |
| `id` | Unique identifier for each data point. |
| `age` | Age of the individual, categorized in 5-year intervals. |
| `height(cm)` | Height of the individual in centimeters. |
| `weight(kg)` | Weight of the individual in kilograms. |
| `waist(cm)` | Waist circumference of the individual in centimeters. |
| `eyesight(left)` | Eyesight measurement for the left eye. |
| `eyesight(right)` | Eyesight measurement for the right eye. |
| `hearing(left)` | Hearing ability for the left ear, represented as binary (`1` or `2` / `0` or `1`). |
| `hearing(right)` | Hearing ability for the right ear, represented as binary (`1` or `2` / `0` or `1`). |
| `systolic` | Systolic blood pressure measurement. |
| `relaxation` | Diastolic blood pressure measurement. |
| `fasting blood sugar` | Fasting blood sugar level. |
| `Cholesterol` | Total cholesterol level. |
| `triglyceride` | Triglyceride level. |
| `HDL` | High-density lipoprotein cholesterol level. |
| `LDL` | Low-density lipoprotein cholesterol level. |
| `hemoglobin` | Hemoglobin level in the blood. |
| `Urine protein` | Level of protein in urine, categorized. |
| `serum creatinine` | Serum creatinine level. |
| `AST` | Level of aspartate aminotransferase enzyme. |
| `ALT` | Level of alanine aminotransferase enzyme. |
| `Gtp` | Level of gamma-glutamyl transferase enzyme. |
| `dental caries` | Presence (`1`) or absence (`0`) of dental cavities. |
| `smoking` | **Target Variable**: Indicates if the individual is a smoker (`1`) or not (`0`). *(Not present in the test set; evaluated on Kaggle)* |

---

## Assessment Components & Grading Breakdown

Your final grade for this project is evaluated across four core components (plus an optional bonus):

| Component | Format / Deliverable | Platform | Evaluation Criteria |
| :--- | :--- | :--- | :--- |
| **1. Kaggle Score** | Leaderboard Standing (`submission.csv`) | Kaggle Competition | Metric: **ROC AUC** |
| **2. Code Submission** | Python Notebook (`.ipynb` or `.py`) | MyConnexion | Code quality, pipeline completeness, reproducibility |
| **3. Learning Log** | Written Reflection (`.md`) | MyConnexion | Depth of insights, conceptual understanding, authentic reflection |
| **4. 5-Min Presentation** | Presentation to Lecturer & Q&A | In-Class / In-Person | Key takeaways, insights, lessons learned |
| **⭐ Bonus Component** | GitHub Repository (`ML`) | GitHub (Public Link) | Proper version control setup with code and `.md` learning log |

---

### 1. Kaggle Competition Score
* Your score on the **Kaggle Leaderboard** evaluates model performance using the **Area Under the ROC Curve (ROC AUC)** metric.
* Scores are computed on predicted probabilities ranging continuously from `0.0` to `1.0`.

### 2. Final Code Submission (via MyConnexion)
* Upload your clean, fully executed Jupyter Notebook (`.ipynb`) or Python script (`.py`) to **MyConnexion**.
* **Key Requirements:**
  * Clean structure with descriptive markdown headers and explanatory comments.
  * Clear evidence of the full ML lifecycle: Exploratory Data Analysis (EDA), preprocessing, feature engineering, model training, hyperparameter tuning, and prediction generation.
  * **Reproducibility:** The notebook must execute from top to bottom without errors.

### 3. Individual Learning Log (via MyConnexion)
* Submit a written reflection in **Markdown format (`.md`)** detailing your personal machine learning journey, technical experiments, and takeaways throughout this project.

> [!TIP]
> ### Pro-Tip: Using GenAI as Your Personal ML Tutor
> You are encouraged to use Generative AI tools (e.g., ChatGPT, Claude, Gemini) as an interactive learning tutor!
> * **Prompt with curiosity:** At each step of your code, ask GenAI to teach you *why* a particular method is used (e.g., *"Why use RobustScaler instead of StandardScaler here?"*, *"Explain how LightGBM handles categorical features step by step."*).
> * **Write in your own words:** Synthesize what you learned and write your log in **your own words**. Personalized explanations, notes on what surprised you, and reflections on your mistakes will score much better than copy-pasting raw AI output.

* **Guiding Pointers for Your Learning Log:**
  1. **Data Understanding & Cleaning:** What patterns or anomalies did you uncover during EDA? How did you handle missing values, skewed distributions, outliers, or feature scaling?
  2. **Model Selection & Experimentation:** Which algorithms did you evaluate (e.g., Logistic Regression, Decision Trees, Random Forests, XGBoost, LightGBM)? How did their validation scores compare?
  3. **Continuous Probability vs. Hard Classification:** What did you learn about predicting continuous probabilities (`predict_proba`) versus discrete labels (`predict`) for the ROC AUC metric?
  4. **Roadblocks & Debugging:** What was the biggest hurdle or bug you encountered (e.g., data leakage, overfitting, submission formatting issues), and how did you diagnose and overcome it?
  5. **External Data Experiments (Optional):** Did you test augmenting your training with the original bio-signals dataset? What impact did it have on cross-validation vs. Kaggle leaderboard scores?

### 4. 5-Minute Presentation
* Prepare a short **5-minute presentation** to present to your lecturer.
* **Focus Area:** Keep it focused and impactful. Rather than walking through every line of code, highlight your **most valuable insights, surprises, and key lessons** (e.g., which features mattered most, what failed and why, or key takeaways about model tuning).

---

## Bonus Opportunity: Version Control with GitHub

You can earn **bonus marks** by adopting industry-standard version control practices for your project deliverables!

> [!TIP]
> ### 💡 Best Practice: Set Up Early & Commit Often
> Set up your repository **before you start coding**. Whenever you complete an EDA step, test a model, or write a log entry:
> * **Commit** with a clear message (e.g., `Add EDA correlation plots`, `Update learning log with scaling reflections`).
> * **Push** to GitHub regularly.
>
> Committing frequently creates a clear timeline of your learning journey, protects your work from accidental loss, and demonstrates your personal development process.

### Steps to Earn the Bonus:
1. **Create a GitHub Account:** Sign up at [github.com](https://github.com) if you do not already have an account.
2. **Download GitHub Desktop:** Install [GitHub Desktop](https://desktop.github.com/) to manage and sync your repository through a simple visual interface.
3. **Create a Repository:** In GitHub Desktop, click `File > New Repository...` and name it `MachineLearning` (or `my-ML-project`).
4. **Push Your Files:** Save your project files in the repository folder, then commit and push:
   * Your project code notebook or script (`.ipynb` / `.py`)
   * Your individual learning log (`.md`)
5. **Submit the Link:** Ensure your repository is set to **Public**, and provide your **GitHub repository URL** with your MyConnexion submission (or in your learning log header).

### 📖 Beginner Tutorial: GitHub Desktop
If you are new to GitHub Desktop, check out this official, simple quickstart guide:
* 👉 **[Getting Started with GitHub Desktop (Official Guide)](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop)** — Step-by-step guide on creating a repo, making commits with messages, and pushing/pulling changes in just a few clicks.

---

## Submission Cheat Sheet (Kaggle Export)

> [!IMPORTANT]
> Kaggle evaluates submissions using **ROC AUC**. Your submission file **MUST** contain continuous probabilities (`0.0` to `1.0`) from `model.predict_proba()[:, 1]`, **NOT** discrete `0` or `1` labels from `model.predict()`.

Below is a sample code snippet to help you format and export your predictions into the correct Kaggle submission template:

```python
import pandas as pd

# 1. Load test data 
test_df = pd.read_csv('test.csv')
X_test = test_df.drop(columns=['id'])

# 2. Predict continuous PROBABILITIES (Column index 1 = Smoker / Positive Class)
# Ensure you use .predict_proba(), NOT .predict()
test_probabilities = model.predict_proba(X_test)[:, 1]

# 3. Format output DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'smoking': test_probabilities
})

# 4. Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ Submission file ready! Verify that values under 'smoking' are continuous probabilities (0.0 to 1.0).")
```

---

## Submission & Deliverables Summary

| Deliverable | Platform | Accepted File Format | Description |
| :--- | :--- | :--- | :--- |
| **Kaggle Leaderboard Score** | Kaggle Competition | `.csv` (`submission.csv`) | Continuous predicted probabilities for the test set |
| **Final Code Notebook** | MyConnexion | `.ipynb` or `.py` | Fully executed, clean, and reproducible code |
| **Learning Log Report** | MyConnexion | `.md` (Markdown) | Structured personal reflection & experiment notes |
| **5-Minute Presentation** | In-Class / In-Person | Presentation Deck / Slides | Key insights, lessons learned, and Q&A |
| **⭐ [Bonus] GitHub Repository** | MyConnexion / Learning Log | Public URL link | GitHub repo under `ML` containing your code and `.md` log |