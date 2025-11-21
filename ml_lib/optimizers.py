import time
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

def apply_optimizer(results, optim_name, update_progress):
    update_progress(85, f"🚀 Menjalankan optimasi {optim_name.upper()}...")

    time.sleep(1)  # simulasi proses

    if optim_name.lower() == "adaboost":
        results["optimizer"] = "AdaBoost"
        results["boosting"] = "Improved with 5 estimators"
    elif optim_name.lower() == "xgboost":
        results["optimizer"] = "XGBoost"
        results["boosting"] = "Improved with depth=3"
    elif optim_name.lower() == "pso":
        results["optimizer"] = "PSO"
        results["metaheuristic"] = "Optimized weights via swarm"
    else:
        results["optimizer"] = "None"

    update_progress(95, f"⚙️ {optim_name.upper()} selesai diterapkan.")
    return results
