Python 3.9.19 (main, May  6 2024, 20:12:36) [MSC v.1916 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 8.15.0 -- An enhanced Interactive Python.


"""
@author: cheikh 
12 Projet ML
"""
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier


data = pd.read_excel(r'C:\Users\cheik\Downloads\ML Health.xlsx', sheet_name='ML Health')

numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                    'DiabetesPedigreeFunction', 'Age']
target = 'Outcome'


numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                    'DiabetesPedigreeFunction', 'Age']

plt.figure(figsize=(15, 10))

for i, feature in enumerate(numeric_features, 1):
    plt.subplot(3, 3, i)
    plt.hist(data[feature], bins=30, edgecolor='black')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()

plt.show()

![Figure_1](https://github.com/user-attachments/assets/fad3458e-0edc-455b-bb4c-c8bdc4537f80)

L'analyse des distributions des caractéristiques liées au diabète révèle plusieurs tendances significatives. Les données sur les grossesses montrent que la majorité des femmes ont eu entre 0 et 5 grossesses, avec très peu de cas dépassant 10 grossesses. Le taux de glucose suit une distribution normale centrée entre 100-150 mg/dL, tandis que la pression artérielle présente également une distribution normale avec un pic entre 70-80 mmHg, bien que quelques valeurs anormalement basses suggèrent de possibles erreurs de mesure. L'épaisseur cutanée et l'insuline présentent des distributions problématiques avec un nombre important de valeurs à 0, indiquant probablement des données manquantes. L'IMC suit une distribution normale centrée autour de 30-35, ce qui indique une prévalence du surpoids dans l'échantillon. La fonction pédigrée du diabète montre une distribution asymétrique avec la majorité des valeurs entre 0 et 1, et l'âge des patients est principalement concentré entre 20 et 40 ans, avec une diminution progressive après 50 ans. Ces observations soulignent l'importance d'un prétraitement approprié des données, particulièrement pour les variables présentant des valeurs manquantes, avant de procéder à la modélisation prédictive du diabète.

# Diagramme en barres pour la répartition des classes

plt.figure(figsize=(6, 4))

data['Outcome'].value_counts().plot(kind='bar', color=['lightblue', 'lightgreen'])

plt.title('Répartition des Classes (Outcome)')

plt.xlabel('Outcome')

plt.ylabel('Compte')

plt.xticks(rotation=0)

plt.show

(![Figure_2](https://github.com/user-attachments/assets/6f7129c4-c7ce-416f-96a4-d5c85c2cd380)

Le nuage de points comparant le taux de glucose et l'IMC révèle une distinction intéressante entre les patients diabétiques (Outcome 1) et non diabétiques (Outcome 0). On observe une concentration plus élevée de cas diabétiques (points orange) dans la zone de glucose supérieure à 125 mg/dL, ce qui correspond au seuil clinique du diabète. La répartition de l'IMC est relativement similaire entre les deux groupes, majoritairement entre 20 et 45, mais les cas diabétiques semblent légèrement plus présents dans les IMC élevés. Quelques valeurs aberrantes sont visibles, notamment des IMC à 0 qui sont probablement des erreurs de données.

# Matrice de corrélation

plt.figure(figsize=(10, 8))

correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

plt.title('Matrice de Corrélation')

plt.show()

![Figure_3](https://github.com/user-attachments/assets/d519d89d-2761-4dfb-8ac8-4f0404d9c169)

Le diagramme en barres montre clairement un déséquilibre dans la distribution des classes. La classe 0 (non diabétiques) compte environ 500 cas, tandis que la classe 1 (diabétiques) n'en compte qu'environ 270. Ce ratio approximatif de 65%-35% indique un déséquilibre de classes qui pourrait affecter la performance des modèles de prédiction. Cette disparité explique potentiellement pourquoi les modèles ont généralement une meilleure performance sur la classe majoritaire (non diabétiques).
# Nuage de points entre deux caractéristiques

plt.figure(figsize=(8, 6))

for outcome in [0, 1]:
    subset = data[data['Outcome'] == outcome]
    plt.scatter(subset['Glucose'], subset['BMI'], label=f'Outcome {outcome}')
plt.title('Glucose vs BMI par Outcome')

plt.xlabel('Glucose')

plt.ylabel('BMI')

plt.legend()

plt.show()

![Figure_4](https://github.com/user-attachments/assets/74ab77ca-b2bc-49e2-b22e-a773f93747bc)

La matrice de corrélation révèle plusieurs relations importantes entre les variables. Le glucose montre la plus forte corrélation positive avec l'outcome (0.47), ce qui en fait le meilleur prédicteur individuel du diabète. L'âge et le nombre de grossesses sont fortement corrélés entre eux (0.54), suggérant une relation naturelle entre ces variables. L'IMC présente une corrélation modérée avec l'outcome (0.29), tandis que l'épaisseur cutanée et l'insuline montrent une corrélation significative entre elles (0.44). La pression artérielle et la fonction pédigrée du diabète montrent des corrélations relativement faibles avec les autres variables, ce qui suggère qu'elles pourraient apporter des informations complémentaires uniques au modèle.
# Prétraitement des données

X = data[numeric_features]

y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = Pipeline(steps=[('scaler', StandardScaler())])

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

# Entrainement

for name, model in models.items():
    model.fit(X_train, y_train)    
   # Prédiction
   y_pred = model.predict(X_test)
    
   # Évaluation
  print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print("\n")

--- Logistic Regression ---

Accuracy: 0.7532467532467533

              precision    recall  f1-score   support
           0       0.81      0.80      0.81        99
           1       0.65      0.67      0.66        55
    accuracy                           0.75       154   
   macro avg       0.73      0.74      0.73       154
weighted avg       0.76      0.75      0.75       154



--- Random Forest ---

Accuracy: 0.7207792207792207

              precision    recall  f1-score   support
           0       0.79      0.78      0.78        99
           1       0.61      0.62      0.61        55     
    accuracy                           0.72       154
                                                          macro avg       0.70      0.70      0.70       154
                                                       weighted avg       0.72      0.72      0.72       154



--- Support Vector Machine ---

Accuracy: 0.7337662337662337

              precision    recall  f1-score   support

           0       0.77      0.83      0.80        99
           1       0.65      0.56      0.60        55
    accuracy                           0.73       154
                                                         macro avg       0.71      0.70      0.70       154
                                                      weighted avg       0.73      0.73      0.73       154



--- Neural Network ---

Accuracy: 0.7077922077922078

              precision    recall  f1-score   support
           0       0.78      0.76      0.77        99
           1       0.59      0.62      0.60        55      
    accuracy                           0.71       154
                                                         macro avg       0.68      0.69      0.69       154
                                                      weighted avg       0.71      0.71      0.71       154



--- XGBoost ---

Accuracy: 0.7077922077922078

              precision    recall  f1-score   support
           0       0.79      0.74      0.76        99
           1       0.58      0.65      0.62        55
    accuracy                           0.71       154
                                                       macro avg       0.69      0.70      0.69       154
                                                    weighted avg       0.72      0.71      0.71       154


Dans cette analyse de modèles de machine learning pour prédire le diabète, la régression logistique se distingue avec la meilleure performance globale, atteignant 75,32% de précision dans la détection des cas de diabète. Les modèles montrent une capacité à différencier les patients diabétiques et non diabétiques, mais avec des variations de performance. La classe 0 (patients sans diabète) est systématiquement mieux prédite que la classe 1 (patients diabétiques), ce qui pourrait indiquer des difficultés à identifier précisément les cas de diabète.

