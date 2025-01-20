import requests
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace

def load_data(ref_path: str, prod_path: str):
    ref_data = pd.read_csv(ref_path)
    prod_data = pd.read_csv(prod_path)
    return ref_data, prod_data

if __name__ == "__main__":
    # Chargement des données
    ref_data, prod_data = load_data("../data/ref_data_with_prediction.csv", "../data/prod_data.csv")

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
                # probas_threshold=0.5,       # Si vous aviez des probabilités brutes, on binarise à 0.5
                # k=3                         # Exemple: on peut étudier la performance top-k
            ),
        ]
    )

    report.run(reference_data=ref_data, current_data=prod_data)
    report_path = "/reports/evidently_report.html"
    report.save_html(report_path)
    print(f"Rapport Evidently généré dans {report_path}")

    # Publier le projet et le rapport via l'API d'Evidently
    # api_url = "http://localhost:8082/api/projects"
    # project_payload = {"name": "Emotion Classification Report"}

    # response = requests.post(api_url, json=project_payload)
    # if response.status_code == 200:
    #     project_id = response.json()["id"]
    #     print(f"Projet créé avec succès : {project_id}")

    #     # Ajouter un rapport
    #     upload_url = f"http://localhost:8082/api/projects/{project_id}/reports"
    #     with open(report_path, "rb") as f:
    #         report_response = requests.post(upload_url, files={"file": f})

    #     if report_response.status_code == 200:
    #         print("Rapport ajouté au projet avec succès.")
    #     else:
    #         print(f"Erreur lors de l'ajout du rapport : {report_response.text}")
    # else:
    #     print(f"Erreur lors de la création du projet : {response.text}")