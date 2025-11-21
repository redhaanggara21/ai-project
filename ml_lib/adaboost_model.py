import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

def train_adaboost(dataset_name, update_progress):
    """
    Training AdaBoost untuk data tabular (mirip C45, tapi boosting).
    Dataset: dataset/{dataset_name}/data.csv
    """

    file_path = os.path.join("dataset", dataset_name, "data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    update_progress(10, "📂 Membaca dataset...")

    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    update_progress(30, "✂️ Membagi data train/test...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    update_progress(50, "🧠 Melatih model AdaBoost (base: Decision Tree)...")

    base = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
    model = AdaBoostClassifier(base_estimator=base, n_estimators=100, learning_rate=0.5, random_state=42)
    model.fit(X_train, y_train)

    update_progress(80, "📈 Menghitung akurasi...")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    model_path = os.path.join("dataset", dataset_name, "adaboost_model.joblib")
    dump(model, model_path)

    update_progress(100, "✅ Training AdaBoost selesai.")

    return {
        "algorithm": "AdaBoost",
        "dataset": dataset_name,
        "accuracy": float(acc),
        "model_path": model_path,
        "num_features": len(X.columns),
        "num_samples": len(data),
    }
