import os
import threading
import numpy as np
import joblib
import traceback
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import xgboost as xgb

# ========================
# Config / Paths
# ========================
MODEL_DIR = getattr(settings, "MODEL_DIR", os.path.join(settings.BASE_DIR, "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

cnn_path = os.path.join(MODEL_DIR, "cnn_model.h5")
xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")

# Try load pre-existing models (optional)
cnn_model = load_model(cnn_path) if os.path.exists(cnn_path) else None
xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None

# Default class names (will be replaced by dataset class order)
class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Training status shared with frontend
training_status = {
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0,
    "status": "idle",
    "metrics": {},
    "scatter_features": []
}
stop_training_flag = False

# ========================
# Helpers
# ========================
def safe_int_division(a, b):
    try:
        return int((a / b) * 100) if b and b > 0 else 0
    except Exception:
        return 0

def update_training_progress(**kwargs):
    global training_status
    training_status.update(kwargs)
    epoch = training_status.get("epoch", 0)
    total = training_status.get("total_epochs", 1)
    training_status["progress"] = safe_int_division(epoch, total)

# ========================
# Start Training Endpoint
# ========================
@csrf_exempt
def start_training(request):
    """
    Start training in a background thread.
    Behavior:
    - If single dataset dir present, use validation_split.
    - Else, expect train/val subfolders.
    - Data augmentation is added.
    - Use embeddings to train XGBoost.
    - Export confusion matrix and PCA-sampled scatter points.
    """
    global stop_training_flag, cnn_model, xgb_model, class_names

    if training_status["status"] == "training":
        return JsonResponse({"message": "Training sedang berjalan!"})

    stop_training_flag = False
    total_epochs = 5
    training_status.update({"status": "training", "epoch": 0, "total_epochs": total_epochs, "metrics": {}, "scatter_features": []})

    # possible dataset locations
    single_dataset_dir = os.path.join(settings.BASE_DIR, "dataset/retina")  # if single folder with subfolders per class
    train_dir = os.path.join(settings.BASE_DIR, "dataset/retina/train")
    val_dir = os.path.join(settings.BASE_DIR, "dataset/retina/val")

    def train_thread():
        global cnn_model, xgb_model, class_names, stop_training_flag
        try:
            # Decide dataset loading strategy
            use_split = False
            if os.path.exists(single_dataset_dir) and any(os.path.isdir(os.path.join(single_dataset_dir, d)) for d in os.listdir(single_dataset_dir)):
                # If there are class subfolders under dataset/retina, use validation_split
                dataset_dir = single_dataset_dir
                use_split = True
            elif os.path.exists(train_dir) and os.path.exists(val_dir):
                use_split = False
            else:
                training_status["status"] = "error"
                training_status["metrics"] = {"error": "Dataset not found. Place data in dataset/retina (class subfolders) or dataset/retina/train & /val."}
                return

            img_size = (128, 128)
            batch_size = 16
            seed = 42

            if use_split:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_dir,
                    validation_split=0.2,
                    subset="training",
                    seed=seed,
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=True
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    dataset_dir,
                    validation_split=0.2,
                    subset="validation",
                    seed=seed,
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=False
                )
            else:
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    train_dir,
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=True
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    val_dir,
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=False
                )

            # sync class names from dataset
            class_names = list(train_ds.class_names)

            # Data augmentation
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.08),
            ])

            # Build model
            cnn_model = tf.keras.Sequential([
                data_augmentation,
                tf.keras.layers.Rescaling(1.0 / 255, input_shape=(img_size[0], img_size[1], 3)),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(name="flatten_layer"),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(class_names), activation='softmax')
            ])

            cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten_layer").output)

            # Optionally cache datasets for faster repeated epochs
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds_cached = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds_cached = val_ds.prefetch(buffer_size=AUTOTUNE)

            # Training loop
            for epoch in range(total_epochs):
                if stop_training_flag:
                    training_status["status"] = "stopped"
                    return

                history = cnn_model.fit(train_ds_cached, validation_data=val_ds_cached, epochs=1, verbose=0)
                training_status["epoch"] = epoch + 1
                training_status["loss"] = float(history.history.get('loss', [0.0])[0])
                training_status["accuracy"] = float(history.history.get('accuracy', [0.0])[0])

                # Efficient validation predictions: predict on whole val_ds once
                try:
                    val_preds = cnn_model.predict(val_ds_cached, verbose=0)
                    y_pred = np.argmax(val_preds, axis=1)
                    # collect y_true
                    y_true = np.concatenate([y.numpy() for _, y in val_ds_cached], axis=0)
                except Exception:
                    # fallback to batch loop if something unexpected happens
                    y_true, y_pred = [], []
                    for imgs, labels in val_ds_cached:
                        preds = np.argmax(cnn_model.predict(imgs, verbose=0), axis=1)
                        y_true.extend(labels.numpy().tolist())
                        y_pred.extend(preds.tolist())
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)

                # Metrics
                try:
                    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                except Exception:
                    precision = recall = f1 = 0.0

                # Confusion matrix NxN
                try:
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
                    cm_list = cm.tolist()
                except Exception:
                    cm_list = [[0] * len(class_names) for _ in range(len(class_names))]

                training_status["metrics"] = {
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "confusion_matrix": cm_list,
                    "class_labels": class_names
                }

                # Build embeddings on a *sampled subset* of training set to avoid huge payloads
                embeddings_list, labels_list = [], []
                max_embedding_points = 600  # configurable
                for imgs_batch, lbls_batch in train_ds:
                    emb = embedding_model.predict(imgs_batch, verbose=0)
                    embeddings_list.append(emb)
                    labels_list.extend(lbls_batch.numpy().tolist())

                if embeddings_list:
                    embeddings = np.vstack(embeddings_list)
                    labels_arr = np.array(labels_list)

                    # sample if too many
                    if embeddings.shape[0] > max_embedding_points:
                        idx = np.random.choice(embeddings.shape[0], max_embedding_points, replace=False)
                        embeddings_sampled = embeddings[idx]
                        labels_sampled = labels_arr[idx]
                    else:
                        embeddings_sampled = embeddings
                        labels_sampled = labels_arr

                    # PCA to 3 dims
                    n_samples, n_features = embeddings_sampled.shape
                    n_components = min(3, n_samples, n_features)
                    if n_components >= 1:
                        pca = PCA(n_components=n_components)
                        scatter_3d_raw = pca.fit_transform(embeddings_sampled)
                        if n_components < 3:
                            pad = np.zeros((scatter_3d_raw.shape[0], 3 - n_components))
                            scatter_3d = np.hstack([scatter_3d_raw, pad])
                        else:
                            scatter_3d = scatter_3d_raw
                    else:
                        scatter_3d = np.zeros((n_samples, 3))
                else:
                    scatter_3d = np.zeros((0, 3))
                    labels_sampled = np.array([])

                scatter_features = []
                for i in range(scatter_3d.shape[0]):
                    lbl_idx = int(labels_sampled[i]) if i < len(labels_sampled) else 0
                    lbl_name = class_names[lbl_idx] if 0 <= lbl_idx < len(class_names) else "UNKNOWN"
                    scatter_features.append({
                        "x": float(scatter_3d[i, 0]),
                        "y": float(scatter_3d[i, 1]),
                        "z": float(scatter_3d[i, 2]),
                        "label": lbl_name
                    })

                training_status["scatter_features"] = scatter_features
                update_training_progress(epoch=training_status["epoch"], total_epochs=total_epochs,
                                         loss=training_status["loss"], accuracy=training_status["accuracy"])

            # Save CNN model
            try:
                cnn_model.save(cnn_path)
            except Exception as e:
                print("Warning: gagal menyimpan cnn model:", e)

            # Train XGBoost on *all* embeddings from train_ds (use embedding_model)
            try:
                # collect ALL embeddings (may be heavy; if dataset huge, sample or accumulate in chunks)
                all_embeddings, all_labels = [], []
                for imgs_batch, lbls_batch in train_ds:
                    emb = embedding_model.predict(imgs_batch, verbose=0)
                    all_embeddings.append(emb)
                    all_labels.extend(lbls_batch.numpy().tolist())

                if all_embeddings:
                    X_emb = np.vstack(all_embeddings)
                    y_emb = np.array(all_labels)

                    # Train XGBoost with some sensible defaults
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        use_label_encoder=False,
                        eval_metric='mlogloss',
                        subsample=0.8
                    )
                    xgb_model.fit(X_emb, y_emb)
                    joblib.dump(xgb_model, xgb_path)
            except Exception as e:
                print("Warning: gagal train/save xgb:", e)

            training_status["status"] = "done"

        except Exception as e:
            training_status["status"] = "error"
            training_status["metrics"] = training_status.get("metrics", {})
            training_status["metrics"]["error"] = str(e)
            print("❌ Error saat training:", e)
            traceback.print_exc()

    t = threading.Thread(target=train_thread, daemon=True)
    t.start()
    return JsonResponse({"message": "Training dimulai!"})


# ========================
# Stop Training
# ========================
@csrf_exempt
def stop_training(request):
    global stop_training_flag
    stop_training_flag = True
    return JsonResponse({"message": "Stop training dikirim!"})


# ========================
# Get Progress
# ========================
def get_progress(request):
    status = training_status.copy()

    # default metrics if missing
    if "metrics" not in status or not status["metrics"]:
        n = len(class_names)
        status["metrics"] = {
            "confusion_matrix": [[0 for _ in range(n)] for _ in range(n)],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "class_labels": class_names
        }
    else:
        status["metrics"].setdefault("class_labels", class_names)

    status.setdefault("scatter_features", [])
    status["progress"] = min(max(status.get("progress", 0), 0), 100)
    return JsonResponse(status)


# ========================
# Download Model
# ========================
def download_model(request, model_type):
    if model_type == "cnn":
        path = cnn_path
    elif model_type == "xgb":
        path = xgb_path
    else:
        return JsonResponse({"error": "Invalid model type"})

    if os.path.exists(path):
        return FileResponse(open(path, "rb"), as_attachment=True)

    return JsonResponse({"error": f"File {model_type} model tidak ditemukan"})
