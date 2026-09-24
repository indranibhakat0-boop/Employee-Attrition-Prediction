# Employee Attrition Prediction using Machine Learning

An end-to-end data analytics and machine learning project that predicts whether an employee
is likely to leave a company (**attrition**), built for the **IBM SkillsBuild – Data Analytics
with AI Academic Internship Program**, conducted by **BharatCares** in association with **AICTE**.

## Project Description

Employee attrition (turnover) is a major cost and planning challenge for organizations. This
project analyzes HR data — demographics, job role, income, satisfaction scores, overtime,
business travel, and tenure — to understand **what drives attrition** and to **predict** which
employees are at risk of leaving.

The pipeline covers:
1. Data generation / loading
2. Exploratory Data Analysis (EDA) with visualizations
3. Data preprocessing (encoding, scaling, train-test split)
4. Model building — Logistic Regression and Random Forest classifiers
5. Model evaluation — accuracy, ROC-AUC, confusion matrix, classification report
6. Feature importance analysis
7. Business insights and conclusions

## Dataset

This project's dataset schema (23 columns, 1,470 employee records) mirrors the well-known
**IBM HR Analytics Employee Attrition & Performance** dataset.

- **Reference dataset (Kaggle):** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Because the project needed to be fully self-contained and reproducible without an external
download step, the notebook **programmatically generates a synthetic dataset with the same
schema and realistic statistical relationships** (e.g., overtime, low job/environment
satisfaction, frequent travel, and lower income increase attrition probability), using a fixed
random seed (`random_state=42`) for reproducibility. The generated data is saved as
[`employee_attrition.csv`](employee_attrition.csv) when the notebook is run. To use the real
Kaggle dataset instead, download it from the link above and replace the data-generation cell
with `pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')`.

**Key columns:** `Age`, `Gender`, `MaritalStatus`, `Department`, `JobRole`, `MonthlyIncome`,
`DistanceFromHome`, `YearsAtCompany`, `OverTime`, `BusinessTravel`, `JobSatisfaction`,
`EnvironmentSatisfaction`, `WorkLifeBalance`, `StockOptionLevel`, `Attrition` (target: Yes/No).

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest) |
| Environment | Jupyter Notebook |

## Project Files

| File | Description |
|---|---|
| `YourName_EmployeeAttritionPrediction.ipynb` | Complete project code (EDA + ML pipeline) |
| `requirements.txt` | Python dependencies required to run the notebook |
| `YourName_ProjectReport.docx` | Full project documentation/report |
| `README.md` | This file |
| `employee_attrition.csv` | Dataset generated when the notebook is run |

## Setup / Run Instructions

1. **Clone or download** this project folder.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
5. Open `YourName_EmployeeAttritionPrediction.ipynb` and run all cells
   (**Cell → Run All**). The notebook generates the dataset, performs EDA, trains both models,
   and displays all evaluation results and plots inline.

## Key Results

- **Random Forest** achieved higher accuracy (~87%) and a solid ROC-AUC (~0.74) compared to
  **Logistic Regression** (~71% accuracy, ROC-AUC ~0.76), with `class_weight='balanced_subsample'`
  used to address class imbalance (~16% attrition rate).
- Top attrition drivers identified: **OverTime, MonthlyIncome, Age, JobSatisfaction,
  EnvironmentSatisfaction, WorkLifeBalance, and BusinessTravel frequency.**
- These findings can help HR teams proactively identify at-risk employees and design targeted
  retention strategies.

## Author

Submitted as part of the **IBM SkillsBuild – Data Analytics with AI Academic Internship
Program**, BharatCares 
