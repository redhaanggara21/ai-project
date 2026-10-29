import os
import threading
import traceback
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# ========================
# Config / Paths
# ========================
MODEL_DIR = os.path.join(settings.BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

cnn_path = os.path.join(MODEL_DIR, "cnn_retina_model.keras")
xgb_path = os.path.join(MODEL_DIR, "xgb_retina_model.pkl")

# Try load pre-existing models (optional)
cnn_model = load_model(cnn_path) if os.path.exists(cnn_path) else None
xgb_model = joblib.load(xgb_path) if os.path.exists(xgb_path) else None

# Default class names
class_names = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Training status shared with frontend
training_status = {
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0,
    "percent": 0,
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
    prog = safe_int_division(epoch, total)
    training_status["progress"] = prog
    training_status["percent"] = prog

# ========================
# Start Training Endpoint
# ========================
@csrf_exempt
def start_training(request):
    global stop_training_flag, cnn_model, xgb_model, class_names

    if training_status["status"] == "training":
        return JsonResponse({"message": "Training sedang berjalan!"})

    stop_training_flag = False
    total_epochs = 5
    training_status.update({
        "status": "training", 
        "epoch": 0, 
        "total_epochs": total_epochs, 
        "metrics": {}, 
        "scatter_features": []
    })

    # Path langsung disesuaikan dengan folder train & val Anda
    train_dir = os.path.join(settings.BASE_DIR, "dataset", "retina", "train")
    val_dir = os.path.join(settings.BASE_DIR, "dataset", "retina", "val")

    def train_thread():
        global cnn_model, xgb_model, class_names, stop_training_flag
        try:
            # Validasi keberadaan folder train & val
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                raise ValueError(f"Folder dataset 'train' atau 'val' tidak ditemukan di: {os.path.join(settings.BASE_DIR, 'dataset', 'retina')}")

            img_size = (128, 128)
            batch_size = 16

            # Memuat dataset langsung dari folder train dan val
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir, image_size=img_size, batch_size=batch_size, shuffle=True
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir, image_size=img_size, batch_size=batch_size, shuffle=False
            )

            class_names = list(train_ds.class_names)

           # Build model CNN 
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.08),
            ])

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

            AUTOTUNE = tf.data.AUTOTUNE
            train_ds_cached = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds_cached = val_ds.prefetch(buffer_size=AUTOTUNE)

            # Training loop per epoch
            for epoch in range(total_epochs):
                if stop_training_flag:
                    training_status["status"] = "stopped"
                    return

                history = cnn_model.fit(train_ds_cached, validation_data=val_ds_cached, epochs=1, verbose=0)
                
                loss_val = float(history.history.get('loss', [0.0])[0])
                acc_val = float(history.history.get('accuracy', [0.0])[0])
                
                training_status["epoch"] = epoch + 1
                training_status["loss"] = loss_val
                training_status["accuracy"] = acc_val

                # Hitung metrik evaluasi per epoch
                val_preds = cnn_model.predict(val_ds_cached, verbose=0)
                y_pred = np.argmax(val_preds, axis=1)
                y_true = np.concatenate([y.numpy() for _, y in val_ds_cached], axis=0)

                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

                training_status["metrics"] = {
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "confusion_matrix": cm.tolist(),
                    "class_labels": class_names
                }

                update_training_progress(epoch=epoch + 1, total_epochs=total_epochs, loss=loss_val, accuracy=acc_val)

            # SIMPAN CNN MODEL SETELAH TRAINING SELESAI
            try:
                cnn_model.save(cnn_path)
                print(f"✅ Model CNN berhasil disimpan di: {cnn_path}")
            except Exception as e:
                print("❌ Warning: gagal menyimpan cnn model:", e)

            # BUAT EMBEDDING MODEL & TRAIN XGBOOST SETELAH CNN SELESAI DILATIH
            try:
                # Karena cnn_model sudah pernah di-fit, layer input sudah memiliki bentuk yang jelas
                embedding_model = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten_layer").output)

                all_embeddings, all_labels = [], []
                for imgs_batch, lbls_batch in train_ds:
                    emb = embedding_model.predict(imgs_batch, verbose=0)
                    all_embeddings.append(emb)
                    all_labels.extend(lbls_batch.numpy().tolist())

                if all_embeddings:
                    X_emb = np.vstack(all_embeddings)
                    y_emb = np.array(all_labels)

                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        eval_metric='mlogloss',
                        subsample=0.8
                    )
                    xgb_model.fit(X_emb, y_emb)
                    joblib.dump(xgb_model, xgb_path)
                    print(f"✅ Model XGBoost berhasil disimpan di: {xgb_path}")
            except Exception as e:
                print("❌ Warning: gagal train/save xgb:", e)

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

    status["percent"] = min(max(status.get("progress", 0), 0), 100)

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

    if status.get("status") == "error":
        err_msg = status["metrics"].get("error", "Terjadi kesalahan saat training.")
        status["message"] = f"❌ Terjadi error: {err_msg}"
    elif status.get("status") == "done":
        status["message"] = "✅ Training selesai!"
    else:
        status["message"] = f"Sedang training epoch {status.get('epoch', 0)}/{status.get('total_epochs', 0)}..."

    status.setdefault("scatter_features", [])
    return JsonResponse(status)


# ========================
# Get Results (Untuk Modal)
# ========================
def get_results(request):
    metrics = training_status.get("metrics", {})
    metrics["accuracy"] = f"{round(training_status.get('accuracy', 0) * 100, 2)}%"
    metrics["scatter_features"] = training_status.get("scatter_features", [])
    return JsonResponse(metrics)


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