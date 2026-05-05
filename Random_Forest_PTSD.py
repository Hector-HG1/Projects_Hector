import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Données
age = np.array([14, 12, 12.5, 47, 51.25, 58, 54.35, 40, 36.8, 58, 14, 37.1, 46, 50, 42.94, 50, 50, 42, 40, 37, 48])
pourcentage_hommes = np.array([55, 7, 20, 80, 25, 75, 84, 69, 64, 85, 91, 78, 64, 75, 70, 83, 78, 68, 74, 78, 100])
nombre_participants = np.array([11, 153, 30, 11, 16, 5, 19, 12, 39, 7, 25, 75, 42, 3, 85, 18, 227, 55, 186, 141, 4])
score_ptsd = np.array([0.34, 0.86, 3.77, 1.03, 0.78, 0.44, 0.73, 4.31, 0.79, 1.16, 0.38, 1, 0.79, 1.64, 0.7, 1.42, 1.09, 1.04, 0.93, 0.65, 0.64])

# Modèle de Random Forest Regressor
modele_forest_reg = RandomForestRegressor()
modele_forest_reg.fit(np.column_stack((age, pourcentage_hommes, nombre_participants)), score_ptsd)

# Prédiction sur les données
predictions_forest = modele_forest_reg.predict(np.column_stack((age, pourcentage_hommes, nombre_participants)))

# Mesures de performance
r2_forest = r2_score(score_ptsd, predictions_forest)
rmse_forest = np.sqrt(mean_squared_error(score_ptsd, predictions_forest))
mae_forest = mean_absolute_error(score_ptsd, predictions_forest)
corr_coef_forest, _ = pearsonr(score_ptsd.flatten(), predictions_forest.flatten())
rmspe_forest = np.sqrt(np.mean(((score_ptsd - predictions_forest) / score_ptsd) ** 2))
nrmse_forest = rmse_forest / (np.max(score_ptsd) - np.min(score_ptsd))
mre_forest = np.mean(np.abs((score_ptsd - predictions_forest) / score_ptsd)) * 100

# Affichage des résultats
print("Modèle de Random Forest Regressor :")
print("R² :", r2_forest)
print("RMSE :", rmse_forest)
print("MAE :", mae_forest)
print("Coefficient de Corrélation (Pearson) :", corr_coef_forest)
print("RMSPE :", rmspe_forest)
print("NRMSE :", nrmse_forest)
print("MRE :", mre_forest)

# Importances des caractéristiques
importances_forest = modele_forest_reg.feature_importances_
print("Importance des caractéristiques (Âge, Pourcentage d'hommes, Nombre de participants) :", importances_forest)

# Calcul des corrélations entre les caractéristiques
corr_age_hommes_forest = np.corrcoef(age, pourcentage_hommes)[0, 1]
corr_age_participants_forest = np.corrcoef(age, nombre_participants)[0, 1]
corr_hommes_participants_forest = np.corrcoef(pourcentage_hommes, nombre_participants)[0, 1]

# Affichage des corrélations
print("Corrélation entre l'âge et le pourcentage d'hommes :", corr_age_hommes_forest)
print("Corrélation entre l'âge et le nombre de participants :", corr_age_participants_forest)
print("Corrélation entre le pourcentage d'hommes et le nombre de participants :", corr_hommes_participants_forest)

# Tracé de la matrice de corrélation
sns.set(font_scale=1.2)
corr_matrix_forest = np.corrcoef(np.column_stack((age, pourcentage_hommes, nombre_participants)), rowvar=False)
feature_names_forest = ['Âge', 'Pourcentage d\'hommes', 'Nombre de participants']
sns.heatmap(corr_matrix_forest, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=feature_names_forest, yticklabels=feature_names_forest)
plt.title('Matrice de corrélation (Random Forest Regressor)')
plt.show()

# Tracer le graphique des importances des caractéristiques
n_features_forest = len(importances_forest)
plt.figure(figsize=(10, 6))
plt.barh(range(n_features_forest), importances_forest, align='center', color='skyblue')
plt.yticks(np.arange(n_features_forest), ['Âge', 'Pourcentage d\'hommes', 'Nombre de participants'])
plt.xlabel('Importance relative')
plt.ylabel('Caractéristiques')
plt.title('Importance des caractéristiques dans le modèle de Random Forest Regressor')
plt.show()

# Tracer le graphique des valeurs réelles vs. valeurs prédites pour toutes les données
plt.scatter(range(len(score_ptsd)), score_ptsd, color='blue', label='Valeurs réelles')
plt.scatter(range(len(predictions_forest)), predictions_forest, color='red', label='Valeurs prédites')
plt.xlabel('Échantillon')
plt.ylabel('Scores PTSD')
plt.title('Valeurs réelles vs. Valeurs prédites (Random Forest Regressor, Toutes les données)')
plt.legend()
plt.show()

# Nouvelles valeurs de test
nouveaux_age = np.array([15, 11, 8, 43]).reshape(-1, 1)
nouveaux_pourcentage_hommes = np.array([0, 0, 60, 100]).reshape(-1, 1)
nouveaux_nombre_participants = np.array([9, 2, 20, 5]).reshape(-1, 1)
nouveaux_score_ptsd = np.array([0.7, 0.54, 0.87, 0.64])

# Tester le modèle sur de nouvelles valeurs
modele_forest_reg_nouveaux = RandomForestRegressor()
modele_forest_reg_nouveaux.fit(np.column_stack((age, pourcentage_hommes, nombre_participants)), score_ptsd)
predictions_nouveaux_forest = modele_forest_reg_nouveaux.predict(np.column_stack((nouveaux_age, nouveaux_pourcentage_hommes, nouveaux_nombre_participants)))
r2_nouveaux_forest = r2_score(nouveaux_score_ptsd, predictions_nouveaux_forest)
rmse_nouveaux_forest = np.sqrt(mean_squared_error(nouveaux_score_ptsd, predictions_nouveaux_forest))
mae_nouveaux_forest = mean_absolute_error(nouveaux_score_ptsd, predictions_nouveaux_forest)
corr_coef_nouveaux_forest, _ = pearsonr(nouveaux_score_ptsd.flatten(), predictions_nouveaux_forest.flatten())
rmspe_nouveaux_forest = np.sqrt(np.mean(((nouveaux_score_ptsd - predictions_nouveaux_forest) / nouveaux_score_ptsd) ** 2))
nrmse_nouveaux_forest = rmse_nouveaux_forest / (np.max(nouveaux_score_ptsd) - np.min(nouveaux_score_ptsd))
mre_nouveaux_forest = np.mean(np.abs((nouveaux_score_ptsd - predictions_nouveaux_forest) / nouveaux_score_ptsd)) * 100

# Affichage des performances sur les nouvelles valeurs
print("\nPerformances sur les nouvelles valeurs (Random Forest Regressor) :")
print("Performance R² sur les nouvelles valeurs :", r2_nouveaux_forest)
print("RMSE sur les nouvelles valeurs :", rmse_nouveaux_forest)
print("MAE sur les nouvelles valeurs :", mae_nouveaux_forest)
print("Coefficient de Corrélation (Pearson) sur les nouvelles valeurs :", corr_coef_nouveaux_forest)
print("RMSPE sur les nouvelles valeurs :", rmspe_nouveaux_forest)
print("NRMSE sur les nouvelles valeurs :", nrmse_nouveaux_forest)
print("MRE sur les nouvelles valeurs :", mre_nouveaux_forest)

# Tracer le graphique des valeurs réelles vs. valeurs prédites uniquement pour les nouvelles valeurs
plt.scatter(range(len(nouveaux_score_ptsd)), nouveaux_score_ptsd, color='blue', label='Valeurs réelles')
plt.scatter(range(len(predictions_nouveaux_forest)), predictions_nouveaux_forest, color='red', label='Valeurs prédites')
plt.xlabel('Échantillon')
plt.ylabel('Scores PTSD')
plt.title('Valeurs réelles vs. Valeurs prédites (Random Forest Regressor, Nouvelles)')
plt.legend()
plt.show()
