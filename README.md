# Electricity-generation-prediction
Electricity generation prediction using machine learning and official government of India data
Electricity Generation Prediction Using Machine Learning


# Electricity Generation Prediction Using Machine Learning

## Project Overview
This project focuses on predicting electricity generation using Machine Learning techniques and government electricity-generation data.

The project applies data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and performance evaluation.

## Objectives
- Analyze electricity generation trends.
- Study state-wise electricity generation.
- Build machine learning models for prediction.
- Compare model performance using evaluation metrics.
- Identify important features affecting electricity generation.

## Dataset
The dataset contains electricity-generation information, including:
- Year and Month
- State and Region
- Installed Capacity (MW)
- Previous Month Generation
- Previous Year Same Month Generation
- Rolling 3-Month Average
- Actual Electricity Generation

**Data Source:**  
- [National Power Portal](https://npp.gov.in/publishedReports)
- [Central Electricity Authority](https://www.cea.nic.in/divisionPage/530?lang=en)
- [Open Government Data Platform](https://data.gov.in/catalog/power-generation)

## Machine Learning Algorithms
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Extra Trees Regressor

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

Run the Python file:

```bash
python Electricity_Generation_Prediction_Complete_Code.py
```

## Project Structure

```text
electricity-generation-prediction/
│
├── Electricity_Generation_Prediction_Complete_Code.py
├── CEA_Statewise_Electricity_Generation_2019-2026.csv
├── README.md
└── .gitignore
```

## Author
Himanshu Kumar

B.Tech – Computer Science and Engineering

Lovely Professional University
