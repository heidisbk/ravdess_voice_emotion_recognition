import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def load_data(ref_path: str, prod_path: str):
    ref_data = pd.read_csv(ref_path)
    prod_data = pd.read_csv(prod_path)
    return ref_data, prod_data

if __name__ == "__main__":
    # Chargement des données
    ref_data, prod_data = load_data("/data/ref_data_with_prediction.csv", "/data/prod_data.csv")

    # Exemple: on suppose que toutes les colonnes "feature_X" sont des features
    feature_cols = [col for col in ref_data.columns if col.startswith("feature_")]

    # Instanciation du rapport Evidently
    # On utilise deux "presets":
    # 1) DataDriftPreset -> analyse le "data drift" sur les features
    # 2) ClassificationPreset -> calcule des métriques de classification
    report = Report(
        metrics=[
            DataDriftPreset(
                columns=feature_cols,         # On ne drift-analyse que les colonnes "feature_..."
                drift_share=0.4,              # Ex: on considère un "dataset drifté" si 40% des features dérivent
                stattest="wasserstein",       # Méthode de test statistique pour comparer les distributions
                stattest_threshold=0.05       # p-value en dessous de laquelle on conclut à un drift
                # cat_stattest etc. si besoin pour des variables catégorielles
            ),
            ClassificationPreset(
                columns=None,               # Par défaut, pas besoin de lister les colonnes
                probas_threshold=0.5,       # Si vous aviez des probabilités brutes, on binarise à 0.5
                # k=3                         # Exemple: on peut étudier la performance top-k
            ),
        ]
    )

    report.run(reference_data=ref_data, current_data=prod_data)
    report.save_html("/data/evidently_report.html")
    print("Rapport Evidently généré dans /data/evidently_report.html")
