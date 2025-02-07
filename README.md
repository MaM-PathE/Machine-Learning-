# 🩺 Projet de Prédiction du Diabète par Apprentissage Automatique

## 🎯 Objectif du Projet
Développer un modèle prédictif capable de détecter précocement le risque de diabète en utilisant des techniques d'apprentissage automatique avancées. L'objectif principal est de créer un outil d'aide au diagnostic qui puisse analyser différents indicateurs de santé et prédire la probabilité de développer un diabète.

## 📊 Jeu de Données
### Source
Fichier Excel : ML Health

### Caractéristiques
- **Caractéristiques Numériques**:
  * Grossesses
  * Niveau de Glucose
  * Tension Artérielle
  * Épaisseur de la Peau
  * Insuline
  * IMC
  * Fonction de Pedigree Diabétique
  * Âge

### Variable Cible
- **Résultat**: Classification binaire (Diabète : Oui/Non)

## 🤖 Modèles d'Apprentissage Automatique
Nous avons implémenté et comparé cinq algorithmes de classification :

1. **Régression Logistique**
   - Modèle linéaire simple et interprétable
   - Entraînement et prédiction rapides
   
2. **Forêt Aléatoire**
   - Méthode d'ensemble avec plusieurs arbres de décision
   - Gère les relations non linéaires
   - Moins sujet au surapprentissage

3. **Machine à Vecteurs de Support (SVM)**
   - Efficace dans les espaces de haute dimension
   - Performant avec une marge de séparation claire

4. **Réseau de Neurones (Perceptron Multi-Couches)**
   - Capture des relations non linéaires complexes
   - Architecture flexible

5. **XGBoost**
   - Algorithme avancé de gradient boosting
   - Haute performance et évolutivité
   - Gère efficacement les interactions entre caractéristiques

## 📈 Métriques d'Évaluation des Modèles
- **Précision**: Performance globale du modèle
- **Précision (Precision)**: Proportion de prédictions positives vraies
- **Rappel (Recall)**: Proportion de positifs réels correctement identifiés
- **Score F1**: Moyenne harmonique de la précision et du rappel

## 🔬 Méthodologie
- Prétraitement des données
- Standardisation des caractéristiques
- Division du jeu de données en ensembles d'entraînement et de test
- Entraînement et évaluation de plusieurs modèles
- Comparaison des performances

## 🌟 Points Clés
- Détection précoce du risque de diabète
- Utilisation de multiples algorithmes de machine learning
- Approche comparative pour identifier le meilleur modèle prédictif

## 🤝 Contributions
1. Forker le dépôt
2. Créer une branche de fonctionnalité
3. Commiter les modifications
4. Pousser la branche
5. Ouvrir une Pull Request

## 📄 Licence
Distribué sous la Licence MIT.
