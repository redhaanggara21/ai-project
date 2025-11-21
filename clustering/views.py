import cv2
import numpy as np
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.shortcuts import render
import tempfile

from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def index(request):
    return render(request, "clustering/index.html")

def kmeans_cluster_image(image_path, n_clusters=5):
    """Cluster image colors using KMeans"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    clustered = kmeans.cluster_centers_.astype("uint8")[labels]
    clustered_img = clustered.reshape(img.shape)
    clustered_img_path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(clustered_img_path, cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR))

    return clustered_img_path, data, labels

def adaboost_classify_clusters(data, labels):
    """Train a simple AdaBoost classifier on KMeans clusters"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        clf = AdaBoostClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        predictions = clf.predict(data)
        return predictions, score
    except Exception as e:
        print(f"[ERROR] AdaBoost classification failed: {e}")
        return labels, 0.0

@csrf_exempt
def cluster_image(request):
    if request.method == "POST" and request.FILES.get("file"):
        try:
            n_clusters = int(request.POST.get("n_clusters", 5))
            if n_clusters < 1 or n_clusters > 50:
                n_clusters = 5
        except:
            n_clusters = 5

        file_path = default_storage.path(default_storage.save(
            request.FILES["file"].name, request.FILES["file"]
        ))
        try:
            # KMeans clustering
            clustered_path, data, labels = kmeans_cluster_image(file_path, n_clusters=n_clusters)

            # AdaBoost classification
            adaboost_labels, score = adaboost_classify_clusters(data, labels)
            # Reconstruct image using AdaBoost labels for demo
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            clustered_adaboost = np.array([np.mean(img.reshape(-1,3)[adaboost_labels==i], axis=0) if np.any(adaboost_labels==i) else [0,0,0] for i in adaboost_labels])
            clustered_adaboost_img = clustered_adaboost.reshape(img.shape).astype("uint8")
            clustered_adaboost_path = tempfile.mktemp(suffix=".png")
            cv2.imwrite(clustered_adaboost_path, cv2.cvtColor(clustered_adaboost_img, cv2.COLOR_RGB2BGR))

            # Return clustered image
            return FileResponse(open(clustered_adaboost_path, "rb"),
                                as_attachment=True,
                                filename=f"clustered_adaboost.png")
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)
