## Diabetes Predictions

Good day everyone! Today we'll be discussing the predictive model we built for Global Diabetes. Our aim is to delve deep into the relationships and predictability between certain health metrics and outcomes.

### Dataset Overview

[Dataset Link](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)

- All patients here are females at least 21 years old of Pima Indian heritage.
- Various metrics included are:
  - Glucose levels
  - Number of pregnancies
  - Blood pressure
  - BMI
  - Age
  - Skin thickness
  - Diabetes pedigree function
  - Insulin levels
  - Outcome (whether or not diabetes was diagnosed)
  
We specifically aimed to predict the 'Outcome' based on these metrics.

### Data Preprocessing

For optimal machine learning performance, we applied MinMaxScaler to scale our data between 0 and 1. This step is critical to ensure that every feature equally influences the outcome prediction.

### Model Selection

While a slew of algorithms were in contention, we benchmarked against the Logistic Regression to set a baseline. Our primary focus, however, remains on two models: 
- Random Forest 
- K Nearest Neighbors (KNN)

#### KNN Predictive Models

```python
# Separate the features, X, from the target variable, y
y = diabetes_df['Outcome']
# Drop features that aren't Glucose, BMI, or Age
X = diabetes_df.drop(columns=['Outcome', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale the data using StandardScalar
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

from sklearn.metrics import classification_report
classification_report = classification_report(y_test, y_pred)
print(classification_report)
```
<img width="522" alt="Screenshot 2023-09-16 at 11 51 01 AM" src="https://github.com/Lalalens/Project-4-Diabetes/assets/127805883/33f12949-58f1-49ef-b1c7-d46d749e919a">
<img width="508" alt="Screenshot 2023-09-16 at 11 51 10 AM" src="https://github.com/Lalalens/Project-4-Diabetes/assets/127805883/c8313156-7a1b-43a7-ba7c-1e3759f2c443">

#### Random Forest

Random Forest, one of our chosen models, functions as an ensemble approach leveraging multiple decision trees for its predictions.

- Test #1 features: Glucose, Insulin, BMI, Age (Achieved 76% accuracy)
- Test #2 features: Pregnancy, Blood Pressure, Age, BMI (Achieved 70% accuracy)
<img width="911" alt="Screenshot 2023-09-16 at 11 51 57 AM" src="https://github.com/Lalalens/Project-4-Diabetes/assets/127805883/a91869e9-6439-43db-ac97-9bb3b10071a0">


#### Linear Regression

To visualize our model's performance, we employed a confusion matrix. This matrix presents the true positives, true negatives, false positives, and false negatives through a heatmap. This visualization is paramount to identify areas where our model might falter.

<img width="638" alt="Screenshot 2023-09-16 at 11 52 27 AM" src="https://github.com/Lalalens/Project-4-Diabetes/assets/127805883/86fdde9e-6840-459f-a111-43ded279d069">


### Evaluation Metrics

For an all-encompassing understanding of our models' efficacy, we scrutinized them using:
- Accuracy
- Precision
- Recall
- F1-score

### Model Performance

Our evaluation revealed the following:
- Random Forest model clinched an impressive accuracy of 76%.

It's important to underline that while both models showcased commendable results, they must be understood in the context of our primary objective - reducing the cost of diabetes medications.

### Conclusion

From data preprocessing to modeling, our journey has endowed us with insights to predict health outcomes from the provided metrics. We're on pins and needles to refine this model further, aiming to drive even higher accuracy and unparalleled value for Global Diabetes.
