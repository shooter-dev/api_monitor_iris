"""
Monitoring Evidently pour détecter le drift et la qualité des données
Compatible avec Evidently 0.7.x
"""
import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Import des métriques Prometheus
from .monitoring_prometheus import (
    DATASET_DRIFT_DETECTED,
    DRIFT_SHARE,
    COLUMN_DRIFT,
    DATA_ROWS_COUNT,
    PREDICTION_CLASS_DISTRIBUTION
)

# Chemins des fichiers
REFERENCE_DATA_PATH = Path("data/reference_data.csv")
CURRENT_DATA_PATH = Path("logfiles/predictions_log.csv")
REPORTS_DIR = Path("evidently_reports")

# Créer le dossier pour les rapports
REPORTS_DIR.mkdir(exist_ok=True)

# Colonnes importantes à surveiller pour le drift
MONITORED_COLUMNS = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']


def load_reference_data() -> pd.DataFrame:
    """Charge les données de référence"""
    df = pd.read_csv(REFERENCE_DATA_PATH)
    return df


def load_current_data(limit: int = 100) -> pd.DataFrame:
    """
    Charge les données de production récentes
    
    Args:
        limit: Nombre de lignes à charger (les plus récentes)
    """
    df = pd.read_csv(CURRENT_DATA_PATH)
    
    # Garder seulement les colonnes nécessaires
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
               'prediction', 'prediction_name']
    
    # Prendre les N dernières lignes
    df = df[columns].tail(limit)
    
    return df


def generate_data_drift_report() -> str:
    """
    Génère un rapport de Data Drift
    Compare les données actuelles avec les données de référence
    
    Returns:
        Chemin du rapport HTML généré
    """
    # Charger les données
    reference = load_reference_data()
    current = load_current_data()
    
    # Créer le rapport avec DataDriftPreset
    report = Report(metrics=[
        DataDriftPreset()
    ])
    
    # Exécuter le rapport et capturer le résultat
    result = report.run(reference_data=reference, current_data=current)
    
    # Sauvegarder en HTML sur l'objet résultat
    report_path = REPORTS_DIR / "data_drift_report.html"
    result.save_html(str(report_path))
    
    return str(report_path)


def generate_data_summary_report() -> str:
    """
    Génère un rapport de résumé/qualité des données
    
    Returns:
        Chemin du rapport HTML généré
    """
    # Charger les données
    reference = load_reference_data()
    current = load_current_data()
    
    # Créer le rapport avec DataSummaryPreset
    report = Report(metrics=[
        DataSummaryPreset()
    ])
    
    # Exécuter le rapport et capturer le résultat
    result = report.run(reference_data=reference, current_data=current)
    
    # Sauvegarder en HTML sur l'objet résultat
    report_path = REPORTS_DIR / "data_summary_report.html"
    result.save_html(str(report_path))
    
    return str(report_path)


def update_prometheus_drift_metrics():
    """
    Met à jour les métriques Prometheus avec les données de drift

    Cette fonction extrait les métriques numériques d'Evidently pour Prometheus.

    Returns:
        dict: Résumé des métriques mises à jour
    """
    # Charger les données
    reference = load_reference_data()
    current = load_current_data()

    # Créer un rapport de drift (utilise le Preset pour compatibilité 0.7.x)
    report = Report(metrics=[DataDriftPreset()])

    # Exécuter le rapport
    result = report.run(reference_data=reference, current_data=current)

    # Extraire les métriques du dictionnaire (Evidently 0.7.x utilise .dict())
    metrics_dict = result.dict()

    summary = {}

    # Parcourir les métriques retournées
    for metric in metrics_dict.get("metrics", []):
        metric_name = str(metric.get("metric_name", ""))

        # Métrique de drift global du dataset
        if "DriftedColumnsCount" in metric_name:
            value_data = metric.get("value", {})
            drifted_count = value_data.get("count", 0)
            drift_share = value_data.get("share", 0.0)

            # Drift détecté si plus de 50% des colonnes ont drifté
            drift_detected = drift_share > 0.5

            # Mettre à jour Prometheus
            DATASET_DRIFT_DETECTED.set(1 if drift_detected else 0)
            DRIFT_SHARE.set(drift_share)

            summary["dataset_drift_detected"] = drift_detected
            summary["drift_share"] = drift_share
            summary["drifted_columns_count"] = drifted_count

        # Drift par colonne
        elif "ValueDrift" in metric_name and "column=" in metric_name:
            # Extraire le nom de la colonne du metric_name
            # Format: "ValueDrift(column=petal_width,method=...)"
            col_name = metric_name.split("column=")[1].split(",")[0]

            # La valeur est la p-value
            p_value = metric.get("value", 1.0)

            # Drift détecté si p-value < 0.05 (seuil par défaut)
            col_drift_detected = p_value < 0.05

            COLUMN_DRIFT.labels(column_name=col_name).set(
                1 if col_drift_detected else 0
            )

    # Compter les lignes
    num_rows = len(current)
    DATA_ROWS_COUNT.set(num_rows)
    summary["num_rows"] = num_rows

    # Distribution des classes de prédiction
    if "prediction_name" in current.columns:
        class_counts = current["prediction_name"].value_counts()
        class_distribution = {}
        for class_name, count in class_counts.items():
            PREDICTION_CLASS_DISTRIBUTION.labels(class_name=class_name).set(count)
            class_distribution[class_name] = int(count)
        summary["class_distribution"] = class_distribution

    return summary


if __name__ == "__main__":
    # Test du module
    print("🔍 Génération du rapport Data Drift...")
    drift_path = generate_data_drift_report()
    print(f"✅ Rapport généré : {drift_path}")

    print("\n🔍 Génération du rapport Data Quality...")
    quality_path = generate_data_summary_report()
    print(f"✅ Rapport généré : {quality_path}")

    print("\n🔍 Mise à jour des métriques Prometheus...")
    summary = update_prometheus_drift_metrics()
    print(f"✅ Métriques mises à jour : {summary}")