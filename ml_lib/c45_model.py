import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import os

def train_c45(dataset_name, update_progress):
    """
    Training algoritma Decision Tree (C4.5)
    Dataset harus berupa file CSV di folder dataset/{dataset_name}/data.csv
    """
    file_path = os.path.join("dataset", dataset_name, "data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    update_progress(10, "📂 Membaca dataset CSV...")

    # Load data
    data = pd.read_csv(file_path)

    if data.shape[1] < 2:
        raise ValueError("Dataset harus memiliki minimal 1 fitur dan 1 label.")

    # Anggap kolom terakhir sebagai label
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    update_progress(25, "📊 Membagi data train/test...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    update_progress(40, "🧠 Melatih model Decision Tree (C4.5)...")

    # Model C4.5 → criterion='entropy'
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)

    update_progress(70, "📈 Menghitung akurasi model...")

    # Evaluasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    update_progress(90, "💾 Menyimpan model dan hasil...")

    # Simpan hasil model (teks rules)
    rules_text = export_text(clf, feature_names=list(X.columns))
    result_path = os.path.join("dataset", dataset_name, "c45_rules.txt")

    with open(result_path, "w") as f:
        f.write(rules_text)

    update_progress(100, "✅ Training C4.5 selesai.")

    return {
        "algorithm": "C4.5 (Decision Tree)",
        "dataset": dataset_name,
        "accuracy": float(acc),
        "model_rules_path": result_path,
        "num_features": len(X.columns),
        "num_samples": len(data),
    }
