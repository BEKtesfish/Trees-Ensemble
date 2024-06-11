
## Heart Disease Prediction using Decision Trees, Random Forest, and XGBoost

### Project Overview

This project involves building machine learning models to predict heart disease using the Heart Failure Prediction Dataset from Kaggle. The models used are Decision Tree, Random Forest, and XGBoost. The goal is to compare the performance of these models and identify the one that provides the highest accuracy in predicting heart disease.

# Dataset

### Source

The dataset is obtained from Kaggle: Heart Failure Prediction Dataset

### Context

Cardiovascular diseases (CVDs) are the leading cause of death globally, accounting for approximately 17.9 million deaths each year, which is 31% of all global deaths. Heart failure is a common outcome of CVDs. Early detection and management are crucial for individuals at high cardiovascular risk.

### Dataset Description

The dataset contains 11 features that can be used to predict the likelihood of heart disease. Below is a summary of the attributes:

* Age: Age of the patient (years)
* Sex: Sex of the patient (M: Male, F: Female)
* ChestPainType: Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
* RestingBP: Resting blood pressure (mm Hg)
* Cholesterol: Serum cholesterol (mg/dl)
* FastingBS: Fasting blood sugar (1 if FastingBS > 120 mg/dl, 0 otherwise)
* RestingECG: Resting electrocardiogram results (Normal, ST: having ST-T wave abnormality, LVH: showing left ventricular hypertrophy)
* MaxHR: Maximum heart rate achieved
* ExerciseAngina: Exercise-induced angina (Y: Yes, N: No)
* Oldpeak: Depression induced by exercise relative to rest
* ST_Slope: The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)
* HeartDisease: Output class (1: heart disease, 0: Normal)


## Data Preprocessing

1. Load the Dataset: The dataset is loaded into a pandas DataFrame.
2. One-Hot Encoding: Categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) are one-hot encoded.
3. Data Splitting: The data is split into training and testing sets (80% train, 20% test).

## Model Training

### Decision Tree

* Model: DecisionTreeClassifier
* Hyperparameter Tuning: GridSearchCV is used to find the best parameters (criterion, min_samples_split, max_depth).
* Results: The best model parameters and accuracy scores for training and testing datasets are printed.

### Random Forest

* Model: RandomForestClassifier
* Hyperparameter Tuning: GridSearchCV is used to tune parameters (criterion, min_samples_split, max_depth, n_estimators).
* Results: The best model parameters and accuracy scores for training and testing datasets are printed.

### XGBoost

* Model: XGBClassifier
* Hyperparameter Tuning: Parameters such as n_estimators and learning_rate are manually set. Early stopping is used to prevent overfitting.
* Results: The model is evaluated on the training and testing datasets, and accuracy scores are printed.

## Evaluation

The accuracy scores for each model are compared to identify the best-performing model. Results indicate that while all models generalize well to the test set, XGBoost offers the highest training accuracy, but its test set accuracy is on par with Random Forest.





### Results

* Decision Tree: Training Accuracy: ~87.87%, Testing Accuracy: ~86.41%
* Random Forest: Training Accuracy: ~88.69%, Testing Accuracy: ~88.59%
* XGBoost: Training Accuracy: ~92.92%, Testing Accuracy: ~88.59%
  
The Random Forest and XGBoost models provide the best generalization performance, with XGBoost showing a slightly higher training accuracy.

Conclusion
The Random Forest and XGBoost models are effective in predicting heart disease with an accuracy of around 88.59% on the test set. Further tuning and additional features could potentially enhance model performance.

References
Kaggle: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
