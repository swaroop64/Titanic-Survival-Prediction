# Titanic Survival Prediction Project

## Project Description

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used for this analysis is the famous Titanic dataset available on Kaggle, which contains information about the passengers, such as age, gender, ticket class, and whether they survived the tragic sinking.

## Dataset

The dataset was sourced from Kaggle: [Titanic Dataset](https://www.kaggle.com/c/titanic/data). It includes the following files:

1. **train.csv**: Training set with features and survival outcomes.
2. **test.csv**: Testing set with features (no survival outcomes).
3. **gender_submission.csv**: Example submission file for the competition.

Key columns in the dataset:
- `PassengerId`: Unique ID for each passenger.
- `Survived`: Survival indicator (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Name`: Passenger's name.
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Passenger fare.
- `Cabin`: Cabin number (if available).
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Objective

The primary objective of this project is to build a machine learning model that predicts whether a passenger survived based on their features. This project demonstrates exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## Project Structure

1. **Exploratory Data Analysis (EDA):**
   - Analyze data distributions.
   - Visualize relationships between features and survival.
   - Handle missing values.

2. **Feature Engineering:**
   - Convert categorical variables to numerical.
   - Engineer new features from existing data.
   - Scale numerical features.

3. **Model Building and Evaluation:**
   - Train multiple machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
   - Evaluate models using metrics like accuracy, precision, recall, and F1 score.
   - Perform hyperparameter tuning to optimize model performance.

4. **Prediction and Submission:**
   - Use the trained model to make predictions on the test dataset.
   - Generate a CSV file for submission to Kaggle.

## Dependencies

- Python (>=3.7)
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Results

The model achieves a survival prediction accuracy of approximately 79%, (80% after optimization) on the test dataset. Key insights and visualizations from the analysis are provided in the notebook.

##

Feel free to explore, modify, and improve the project. Contributions are welcome!
