# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn (leave the company), using classification models trained on customer account, service, and billing data.

## 📌 Problem Statement

Customer retention is more cost-effective than acquiring new customers. This project builds a predictive model to identify customers at high risk of churning, so a business can proactively target them with retention offers.

## 📊 Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Size:** 7,043 customer records, 21 features
- **Target variable:** `Churn` (Yes/No)

## 🛠️ Tech Stack

- Python (Google Colab)
- Pandas, NumPy — data handling
- Matplotlib, Seaborn — visualization
- Scikit-learn — modeling and evaluation

## 🔍 Workflow

1. **Data Cleaning** — handled missing values in `TotalCharges`, removed irrelevant identifiers
2. **Exploratory Data Analysis** — visualized churn distribution, contract type, tenure, and monthly charges against churn
3. **Feature Engineering** — encoded categorical variables using one-hot encoding, converted target to binary
4. **Model Building** — trained Logistic Regression and Random Forest classifiers on an 80/20 train-test split
5. **Evaluation** — compared models using accuracy, precision, recall, and F1-score
6. **Feature Importance** — identified the strongest drivers of churn using the Random Forest model

## 📈 Results

| Model | Accuracy | Churn Recall | Churn Precision |
|---|---|---|---|
| Logistic Regression | 78.75% | 52% | 62% |
| Random Forest | 78.54% | 48% | 63% |

Given the class imbalance in the dataset (~73% No / ~27% Yes), Logistic Regression was selected as the preferred model for this use case due to its higher recall on the churn class — correctly identifying more at-risk customers.

## 💡 Key Business Insights

- **Billing and tenure are the strongest churn predictors** — `TotalCharges`, `MonthlyCharges`, and `tenure` together account for over 50% of the model's predictive power. New customers with high monthly bills are at the highest risk.
- **Fiber optic customers churn more often**, suggesting possible pricing or service quality concerns worth investigating.
- **Contract type matters** — month-to-month customers churn significantly more than those on two-year contracts.
- **Lack of Online Security / Tech Support increases churn risk** — bundling these services could improve retention.

**Recommendation:** Target new, high-bill, month-to-month customers with proactive retention offers such as discounts, contract upgrade incentives, or bundled support services.

## 🚀 How to Run

1. Open the notebook in Google Colab
2. Upload the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset
3. Run all cells in order

## 📂 Files

- `Customer_Churn_Prediction.ipynb` — full analysis and model code
- `README.md` — project overview
