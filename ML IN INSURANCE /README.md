
"""
@author: cheikh 
13 Projet ML
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


data = pd.read_csv(r'C:\Users\cheik\Downloads\TravelInsurancePrediction.csv')

categorical_features = ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']
numeric_features = ['Age', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases']

# Prétraitement des données
X = data.drop('TravelInsurance', axis=1)
y = data['TravelInsurance']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipelines
models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Support Vector Machine': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(random_state=42))
    ]),
    'Neural Network': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
}

# Entraînement
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Prédiction
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print("\n")
--- Logistic Regression ---
Accuracy: 0.7688442211055276
              precision    recall  f1-score   support

           0       0.77      0.92      0.84       257
           1       0.77      0.50      0.60       141

    accuracy                           0.77       398
   macro avg       0.77      0.71      0.72       398
weighted avg       0.77      0.77      0.75       398



--- Random Forest ---
Accuracy: 0.8140703517587939
              precision    recall  f1-score   support

           0       0.82      0.91      0.86       257
           1       0.80      0.64      0.71       141

    accuracy                           0.81       398
   macro avg       0.81      0.77      0.79       398
weighted avg       0.81      0.81      0.81       398



--- Support Vector Machine ---
Accuracy: 0.8065326633165829
              precision    recall  f1-score   support

           0       0.79      0.95      0.86       257
           1       0.86      0.54      0.66       141

    accuracy                           0.81       398
   macro avg       0.83      0.75      0.76       398
weighted avg       0.82      0.81      0.79       398



--- Neural Network ---
Accuracy: 0.7889447236180904
              precision    recall  f1-score   support

           0       0.80      0.89      0.85       257
           1       0.76      0.60      0.67       141

    accuracy                           0.79       398
   macro avg       0.78      0.75      0.76       398
weighted avg       0.79      0.79      0.78       398



--- XGBoost ---
Accuracy: 0.8165829145728644
              precision    recall  f1-score   support

           0       0.82      0.92      0.87       257
           1       0.81      0.62      0.71       141

    accuracy                           0.82       398
   macro avg       0.82      0.77      0.79       398
weighted avg       0.82      0.82      0.81       398


L'analyse des résultats montre que la régression logistique atteint une précision de 0.77, avec un bon rappel pour ceux qui n'ont pas souscrit à l'assurance (classe 0), mais un rappel plus faible pour ceux qui l'ont fait (classe 1). Le Random Forest, avec une précision de 0.81, a une performance globale supérieure, offrant un bon équilibre entre précision et rappel pour les deux classes, particulièrement pour la classe minoritaire (1). La SVM, également avec une précision de 0.81, excelle en précision pour la classe 1 mais manque en rappel. Le réseau neuronal, avec une précision de 0.79, présente un bon équilibre mais n'égale pas le Random Forest en termes de rappel pour la classe 1. En conclusion, le Random Forest est recommandé comme le meilleur modèle pour cette tâche de classification binaire, grâce à sa performance équilibrée et sa capacité à mieux gérer la classe minoritaire.




