# Neurofive ML Track 🚀

This repository contains my tasks and progress for the Neurofive Machine Learning Track.

## Task 1: Exploratory Data Analysis (EDA) on Titanic Dataset
In this task, I performed basic EDA on the famous Titanic dataset to understand its shape, data types, and missing values before moving on to any machine learning models.

### Tools & Libraries Used:
* Python
* Google Colab
* Pandas
* NumPy

### Key Findings:
* **Shape:** The dataset has 891 rows and 12 columns.
* **Missing Values:** `Cabin` has 687 missing values, `Age` has 177, and `Embarked` has 2.
* **Data Types:** A mix of numerical (`Age`, `Fare`) and categorical (`Sex`, `Survived`, `Embarked`) columns.

Check out the Jupyter Notebook (`.ipynb` file) in this repository for the detailed Python code!

## Task 2: Data Cleaning & Exploratory Visualizations
In this task, I prepared the Titanic dataset for machine learning by handling missing values, identifying outliers, and visualizing data to find hidden patterns.

### Key Steps Performed:
* **Data Cleaning:** 
  * Dropped the `Cabin` column as 77% of its data was missing.
  * Filled missing values in `Age` using the median (to avoid outlier influence).
  * Filled missing values in `Embarked` using the mode.
* **Outlier Detection:** Used a boxplot to identify extreme values in the `Fare` column.
* **Data Visualization:** Created 4 key plots using `matplotlib` and `seaborn`:
  1. Boxplot (Fare outliers)
  2. Histogram (Age distribution)
  3. Bar Chart (Survival rates by gender)
  4. Correlation Heatmap (Relationships between numerical features)

### Key Insights & Conclusion:
Visual analysis revealed that **Sex (Gender)** and **Pclass (Passenger Class)** most significantly affected survival. The bar chart showed females having a much higher survival rate, while the heatmap indicated a strong negative correlation (-0.34) between `Pclass` and `Survived` (meaning higher-class passengers were prioritized).

## Task 3: Machine Learning Model (Logistic Regression)
In this task, I built my very first classification model to predict passenger survival based on the cleaned EDA data.

### Key Steps Performed:
* **Feature Encoding:** Converted categorical columns (`Sex`, `Embarked`) into numerical format using One-Hot Encoding (`pd.get_dummies`). Dropped non-predictive text columns.
* **Train/Test Split:** Divided the dataset into 80% training data and 20% testing data.
* **Model Training:** Trained a `LogisticRegression` classification model.
* **Evaluation:** The model achieved an **Accuracy Score of 81.01%**. I also generated a Confusion Matrix to analyze True Positives (55), True Negatives (90), False Positives (15), and False Negatives (19).

### Key Insights:
The model successfully learned the survival patterns, achieving a solid accuracy score. The confusion matrix allowed me to visualize exactly where the model made correct predictions versus where it made false alarms, proving the value of proper data preparation before modeling.
