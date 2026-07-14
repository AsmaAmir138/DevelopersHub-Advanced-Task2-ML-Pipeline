import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Generate Dummy Telco Churn Dataset (No external download needed)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'tenure': np.random.randint(1, 72, n_samples),
    'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
    'TotalCharges': np.random.uniform(18.0, 8000.0, n_samples),
    'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# 2. Define Features and Target
X = data.drop('Churn', axis=1)
y = data['Churn']

# 3. Identify Numerical and Categorical Columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = ['gender', 'Partner', 'SeniorCitizen']

# 4. Create Preprocessing Transformers
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# 5. Build full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. GridSearchCV for Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# 8. Evaluate Best Model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("Best Parameters found:", grid_search.best_params_)
print("\nAccuracy Score:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 9. Save the Pipeline
joblib.dump(best_model, 'churn_pipeline_model.pkl')
print("\nPipeline successfully saved as 'churn_pipeline_model.pkl'!")
