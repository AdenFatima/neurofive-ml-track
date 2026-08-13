
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

## Task 4: House Price Prediction (Linear Regression)
In this task, I shifted from classification to regression by predicting continuous values. I built a Linear Regression model using the California Housing dataset to predict house prices based on selected features like median income and house age.

### Key Steps & Evaluation:
* **Feature Selection:** Selected key features (MedInc, HouseAge, AveRooms, AveOccup) to train the model.
* **Model Training:** Trained a `LinearRegression` model using `scikit-learn`.
* **Evaluation:** Calculated the RMSE (0.81) and R² score (0.50) to measure prediction accuracy.
* **Visualization:** Plotted a scatter chart comparing actual versus predicted prices, showing a solid trend along the perfect prediction line.

## Task 5: Model Evaluation & Hyperparameter Tuning
In this task, I evaluated the Titanic classification model beyond simple accuracy and used hyperparameter tuning to find the best possible settings for the algorithm.

### Key Steps Performed:
* **Advanced Evaluation:** Generated a `classification_report` to analyze Precision, Recall, and F1-score. I learned that accuracy can be misleading in imbalanced datasets because a model could just guess the majority class and score high while failing to detect the minority class.
* **Hyperparameter Tuning:** Used `GridSearchCV` to systematically test different combinations of hyperparameters (`C` and `solver`) for the `LogisticRegression` model.
* **Comparison:** Compared the tuned model's performance with the original default model to observe any improvements in the metrics. 

### Before/After Comparison
| Metric | Original Model | Tuned Model |
| :--- | :--- | :--- |
| **Accuracy** | 81.01% | 78.21% |
| **Best Parameters** | Default (C=1.0, solver='lbfgs') | C=0.1, solver='liblinear' |

**Note on Results:** The tuned model showed a slightly lower accuracy on this specific test set. This is a common occurrence because `GridSearchCV` uses 5-fold cross-validation to find the most generalized and stable parameters across multiple data splits, whereas the default model might have slightly overfitted to our single random train/test split.

## Task 6: Customer Churn Prediction (Business Problem)
In this task, I tackled a real-world business problem by predicting customer churn using the Telco Customer Churn dataset.

### Key Steps & Evaluation:
* **Data Preprocessing:** Cleaned numeric columns (`TotalCharges`) and encoded categorical variables. 
* **Model Comparison:** Trained and compared two models:
    * **Logistic Regression:** Achieved an accuracy of **82.04%**.
    * **Decision Tree Classifier:** Achieved an accuracy of **80.62%**.
* **Business Insights (Feature Importance):** Used the Decision Tree to extract the top 3 features driving customer churn:
    1. `tenure` (How long they stay)
    2. `InternetService_Fiber optic`
    3. `TotalCharges` (Overall billing)
 

## Task 7: Build a Proper ML Pipeline with Feature Engineering
In this task, I elevated my code to industry standards by building a scikit-learn `Pipeline`. I used the Titanic dataset to automate preprocessing and model training.

### Key Steps Performed:
* **Feature Engineering:** Created two new features: `FamilySize` (combining `SibSp` and `Parch`) and `IsAlone` to provide better insights into passenger survival.
* **ColumnTransformer:** Used `SimpleImputer` and `StandardScaler` for numerical columns, and `OneHotEncoder` for categorical columns.
* **Pipeline Creation:** Chained the preprocessing steps with a `LogisticRegression` model into a single, reusable `Pipeline` object.
* **Model Serialization:** Successfully saved the entire pipeline using `joblib` (`titanic_pipeline.pkl`) for future deployment, ensuring that any new raw data fed into the model will automatically undergo the exact same preprocessing steps.

## Task 8: Ensemble Learning (Random Forest vs XGBoost)
In this task, I upgraded from single models to powerful ensemble methods using the Telco Customer Churn dataset.

### Key Steps Performed:
* **Model Training:** Trained a `RandomForestClassifier` and an `XGBClassifier` alongside a baseline `DecisionTreeClassifier`.
* **Feature Importance Analysis:** Visualized and compared the top predictive features. It was fascinating to see how they differ: **Random Forest** prioritized continuous variables like `TotalCharges` and `tenure`, while **XGBoost** heavily prioritized categorical business logic like `InternetService_Fiber optic` and `Contract_Two year`.
* **Algorithm Comparison:** Documented the core architectural difference between parallel ensemble methods (Bagging/Random Forest) and sequential ensemble methods (Boosting/XGBoost).

### Model Performance Comparison
| Model Type | Algorithm | Metric (Accuracy) |
| :--- | :--- | :--- |
| **Baseline (Single)** | Decision Tree | 80.62% |
| **Ensemble (Bagging)** | Random Forest | 78.92% |
| **Ensemble (Boosting)** | XGBoost | 79.42% |
