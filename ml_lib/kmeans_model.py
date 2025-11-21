import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def train_kmeans(dataset_name, update_progress):
    """
    Training algoritma K-Means untuk clustering.
    Dataset bisa berupa:
      - dataset/{dataset_name}/data.csv (tabular)
      - atau folder dataset/{dataset_name}/train/ dengan gambar (fitur CNN diambil manual)
    """

    update_progress(5, "📂 Mengecek dataset...")

    csv_path = os.path.join("dataset", dataset_name, "data.csv")
    image_folder = os.path.join("dataset", dataset_name, "train")

    if os.path.exists(csv_path):
        update_progress(10, "📄 Membaca dataset CSV...")
        data = pd.read_csv(csv_path)
        X = data.values
    elif os.path.exists(image_folder):
        update_progress(10, "🖼️ Membaca dataset gambar...")
        # Ambil fitur sederhana dari gambar (mean pixel) agar cepat
        from PIL import Image
        import glob

        img_files = glob.glob(os.path.join(image_folder, "**/*.jpg"), recursive=True)
        features = []
        for idx, path in enumerate(img_files):
            img = Image.open(path).convert("L").resize((64, 64))
            arr = np.array(img).flatten()
            features.append(arr)
            if idx % 10 == 0:
                update_progress(10 + int(idx / len(img_files) * 50), f"📸 Memproses gambar {idx}/{len(img_files)}")

        X = np.array(features)
        update_progress(60, "📊 Ekstraksi fitur selesai.")
    else:
        raise FileNotFoundError("Dataset tidak ditemukan, pastikan ada folder /data.csv atau /train/")

    update_progress(65, "⚙️ Menjalankan K-Means Clustering...")

    n_clusters = 3  # default cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    update_progress(85, "📈 Menghitung skor siluet...")

    silhouette = silhouette_score(X, labels)

    update_progress(90, "🎨 Membuat visualisasi PCA 2D...")

    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")
    plt.title(f"KMeans Clustering - {dataset_name}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plot_path = os.path.join("dataset", dataset_name, "kmeans_plot.png")
    plt.savefig(plot_path)
    plt.close()

    update_progress(100, "✅ Training K-Means selesai.")

    return {
        "algorithm": "K-Means",
        "dataset": dataset_name,
        "num_samples": len(X),
        "n_clusters": n_clusters,
        "silhouette": float(silhouette),
        "plot_path": plot_path,
    }
