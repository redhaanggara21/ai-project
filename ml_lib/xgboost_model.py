import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from joblib import dump

def train_xgboost(dataset_name, update_progress):
    """
    Training algoritma XGBoost untuk klasifikasi tabular.
    Dataset diambil dari dataset/{dataset_name}/data.csv
    """

    file_path = os.path.join("dataset", dataset_name, "data.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    update_progress(10, "📂 Membaca dataset...")

    df = pd.read_csv(file_path)

    if df.shape[1] < 2:
        raise ValueError("Dataset harus memiliki minimal 1 fitur dan 1 label.")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    update_progress(25, "✂️ Membagi data train/test...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    update_progress(40, "🧠 Melatih model XGBoost...")

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    update_progress(70, "📈 Menghitung akurasi dan laporan...")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    update_progress(90, "💾 Menyimpan model dan hasil...")

    # Simpan model
    model_path = os.path.join("dataset", dataset_name, "xgboost_model.joblib")
    dump(model, model_path)

    # Simpan laporan hasil
    report_path = os.path.join("dataset", dataset_name, "xgboost_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Akurasi: {acc:.4f}\n\n")
        for cls, metrics in report.items():
            f.write(f"{cls}: {metrics}\n")

    update_progress(100, "✅ Training XGBoost selesai.")

    return {
        "algorithm": "XGBoost",
        "dataset": dataset_name,
        "accuracy": float(acc),
        "num_features": len(X.columns),
        "num_samples": len(df),
        "model_path": model_path,
        "report_path": report_path
    }
