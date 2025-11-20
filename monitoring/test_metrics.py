"""
Script pour tester l'API Iris et générer des métriques Prometheus
"""
import requests
import time
import random
from typing import Dict, Any

API_URL = "http://localhost:8000/predict"

# Exemples de données pour chaque classe d'Iris
IRIS_SAMPLES = {
    "setosa": [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
        {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2},
    ],
    "versicolor": [
        {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
        {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
        {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5},
    ],
    "virginica": [
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
        {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9},
        {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1},
    ]
}


def send_prediction(data: Dict[str, float]) -> None:
    """Envoie une requête de prédiction à l'API"""
    try:
        response = requests.post(API_URL, json=data, timeout=5)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prédiction: {result['prediction_name']} (confiance: {result['confidence']:.2%})")
            print(f"   Données: {data}")
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")


def test_all_samples():
    """Teste tous les échantillons prédéfinis"""
    print("=" * 60)
    print("TEST DE TOUTES LES CLASSES D'IRIS")
    print("=" * 60)

    for iris_class, samples in IRIS_SAMPLES.items():
        print(f"\n🌸 Classe: {iris_class.upper()}")
        print("-" * 60)
        for sample in samples:
            send_prediction(sample)
            time.sleep(0.5)  # Petit délai entre les requêtes


def generate_traffic(num_requests: int = 20, delay: float = 1.0):
    """Génère du trafic aléatoire vers l'API"""
    print("\n" + "=" * 60)
    print(f"GÉNÉRATION DE {num_requests} REQUÊTES ALÉATOIRES")
    print("=" * 60)

    all_samples = [sample for samples in IRIS_SAMPLES.values() for sample in samples]

    for i in range(num_requests):
        sample = random.choice(all_samples)
        print(f"\n[{i+1}/{num_requests}]")
        send_prediction(sample)
        time.sleep(delay)


def stress_test(num_requests: int = 50):
    """Test de charge rapide sans délai"""
    print("\n" + "=" * 60)
    print(f"TEST DE CHARGE: {num_requests} REQUÊTES RAPIDES")
    print("=" * 60)

    all_samples = [sample for samples in IRIS_SAMPLES.values() for sample in samples]
    success_count = 0
    error_count = 0

    start_time = time.time()

    for i in range(num_requests):
        sample = random.choice(all_samples)
        try:
            response = requests.post(API_URL, json=sample, timeout=5)
            if response.status_code == 200:
                success_count += 1
                print(f"✅ {i+1}/{num_requests}", end="\r")
            else:
                error_count += 1
                print(f"❌ {i+1}/{num_requests} - Erreur {response.status_code}")
        except Exception as e:
            error_count += 1
            print(f"❌ {i+1}/{num_requests} - Exception: {e}")

    elapsed_time = time.time() - start_time

    print(f"\n\n📊 RÉSULTATS:")
    print(f"   ✅ Succès: {success_count}/{num_requests}")
    print(f"   ❌ Erreurs: {error_count}/{num_requests}")
    print(f"   ⏱️  Temps total: {elapsed_time:.2f}s")
    print(f"   🚀 Requêtes/sec: {num_requests/elapsed_time:.2f}")


if __name__ == "__main__":
    print("🌸 SCRIPT DE TEST API IRIS 🌸\n")

    # Menu
    print("Choisissez une option:")
    print("1. Tester tous les échantillons (9 requêtes)")
    print("2. Générer du trafic aléatoire (20 requêtes avec délai)")
    print("3. Test de charge (50 requêtes rapides)")
    print("4. Tout exécuter")

    choice = input("\nVotre choix (1-4): ").strip()

    if choice == "1":
        test_all_samples()
    elif choice == "2":
        generate_traffic(num_requests=20, delay=1.0)
    elif choice == "3":
        stress_test(num_requests=50)
    elif choice == "4":
        test_all_samples()
        time.sleep(2)
        generate_traffic(num_requests=20, delay=0.5)
        time.sleep(2)
        stress_test(num_requests=50)
    else:
        print("❌ Choix invalide")
        exit(1)

    print("\n" + "=" * 60)
    print("✅ TERMINÉ! Vérifiez vos métriques sur:")
    print("   📊 API Metrics: http://localhost:8000/metrics")
    print("   📈 Prometheus: http://localhost:9090")
    print("   📉 Grafana: http://localhost:3000")
    print("=" * 60)
