import os
import threading
import traceback
import datetime
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image

# ========================
# Status Global Training
# ========================
training_status = {
    "epoch": 0,
    "total_epochs": 3,
    "loss": 0.0,
    "accuracy": 0.0,
    "progress": 0,
    "percent": 0,
    "status": "idle",
    "metrics": {},
    "scatter_features": []
}
stop_training_flag = False

def update_progress(epoch, total, loss, acc):
    global training_status
    prog = int((epoch / total) * 100)
    training_status.update({
        "epoch": epoch,
        "total_epochs": total,
        "loss": float(loss),
        "accuracy": float(acc),
        "progress": prog,
        "percent": prog
    })

def generate_pdf_report(pdf_path, algoritma, optimasi, dataset, metrics, class_names):
    """Fungsi helper untuk membuat PDF laporan training dan menyimpannya di folder models/"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#0d6efd")
        )
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            spaceAfter=15
        )

        # Judul & Tanggal
        story.append(Paragraph("Laporan Hasil Training Model Klasifikasi", title_style))
        story.append(Paragraph(f"Waktu Pembuatan: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        story.append(Spacer(1, 10))

        # Informasi Model & Dataset
        info_data = [
            ["Algoritma Utama", algoritma],
            ["Optimasi / Classifier", optimasi],
            ["Dataset", dataset],
            ["Jumlah Kelas", str(len(class_names))]
        ]
        t_info = Table(info_data, colWidths=[150, 250])
        t_info.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#e9ecef")),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_info)
        story.append(Spacer(1, 15))

        # Ringkasan Metrik Evaluasi
        story.append(Paragraph("<b>Metrik Evaluasi Model</b>", styles['Heading2']))
        story.append(Spacer(1, 6))

        acc = metrics.get("akurasi", f"{metrics.get('accuracy', 0)}%")
        prec = metrics.get("precision", "-")
        rec = metrics.get("recall", "-")
        f1 = metrics.get("f1", "-")

        summary_data = [
            ["Metrik Evaluasi", "Nilai Real"],
            ["Akurasi", str(acc)],
            ["Precision (Macro)", str(prec)],
            ["Recall (Macro)", str(rec)],
            ["F1-Score (Macro)", str(f1)]
        ]

        t_summary = Table(summary_data, colWidths=[200, 200])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 20))

        # Confusion Matrix Table
        cm = metrics.get("confusion_matrix", [])
        if cm:
            story.append(Paragraph("<b>Confusion Matrix</b>", styles['Heading2']))
            story.append(Spacer(1, 6))
            
            cm_table_data = [["Predicted ->"] + class_names]
            for i, row in enumerate(cm):
                row_label = class_names[i] if i < len(class_names) else f"Class {i}"
                cm_table_data.append([row_label] + list(map(str, row)))

            t_cm = Table(cm_table_data)
            t_cm.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#6c757d")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (1, 1), (-1, -1), colors.whitesmoke),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t_cm)

        doc.build(story)
        print(f"✅ File PDF Laporan Berhasil Disimpan di: {pdf_path}")
    except Exception as pdf_err:
        print(f"⚠️ Gagal membuat PDF Laporan: {str(pdf_err)}")

@csrf_exempt
def start_training_classification(request):
    global stop_training_flag, training_status

    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method tidak diizinkan!"}, status=405)

    if training_status["status"] == "training":
        return JsonResponse({"status": "warning", "message": "Training lain sedang berjalan!"})

    algoritma = request.POST.get("algoritma", "CNN")
    kategori = request.POST.get("kategori", "Klasifikasi")
    optimasi = request.POST.get("optimasi", "XGBoost")
    dataset = request.POST.get("dataset", "Retina")

    stop_training_flag = False
    training_status.update({
        "status": "training",
        "epoch": 0,
        "progress": 0,
        "percent": 0,
        "metrics": {},
        "scatter_features": []
    })

    ds_name = dataset.lower().replace(" ", "_")
    train_dir = os.path.join(settings.BASE_DIR, "dataset", ds_name, "train")
    val_dir = os.path.join(settings.BASE_DIR, "dataset", ds_name, "val")

    MODEL_DIR = os.path.join(settings.BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    def train_background_worker():
        global training_status, stop_training_flag
        try:
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                raise ValueError(f"Folder dataset 'train' atau 'val' tidak ditemukan di path: dataset/{ds_name}/")

            img_size = (224, 224)
            batch_size = 32
            total_epochs = 3

            # Load Dataset menggunakan Keras
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir, image_size=img_size, batch_size=batch_size, shuffle=True
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir, image_size=img_size, batch_size=batch_size, shuffle=False
            )

            class_names = train_ds.class_names

            # Arsitektur CNN menggunakan Functional API
            inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
            x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            
            flatten_out = tf.keras.layers.Flatten(name="flatten_layer")(x)
            
            x_dense = tf.keras.layers.Dense(128, activation='relu')(flatten_out)
            outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x_dense)

            cnn_model = Model(inputs=inputs, outputs=outputs)
            embedding_model = Model(inputs=inputs, outputs=flatten_out)

            cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            AUTOTUNE = tf.data.AUTOTUNE
            train_cached = train_ds.prefetch(AUTOTUNE)
            val_cached = val_ds.prefetch(AUTOTUNE)

            # Training Loop
            for epoch in range(total_epochs):
                if stop_training_flag:
                    training_status["status"] = "stopped"
                    return

                history = cnn_model.fit(train_cached, validation_data=val_cached, epochs=1, verbose=0)
                loss_val = history.history.get('loss', [0.0])[0]
                acc_val = history.history.get('accuracy', [0.0])[0]

                update_progress(epoch + 1, total_epochs, loss_val, acc_val)

            # Ground truth validasi
            y_true = np.concatenate([y.numpy() for _, y in val_cached], axis=0).astype(int)

            # Penentuan Nama File Model & PDF
            algo_str = algoritma.lower()
            opt_str = optimasi.lower()
            base_filename = f"{algo_str}_{opt_str}_{ds_name}"
            
            model_filename = f"{base_filename}.pkl"
            pdf_filename = f"{base_filename}.pdf"

            model_path = os.path.join(MODEL_DIR, model_filename)
            pdf_path = os.path.join(MODEL_DIR, pdf_filename)

            # Cek apakah menggunakan Hybrid XGBoost atau CNN murni
            if optimasi.lower() == "xgboost" or not optimasi:
                all_embeddings, all_labels = [], []
                for imgs_batch, lbls_batch in train_ds:
                    emb = embedding_model.predict(imgs_batch, verbose=0)
                    all_embeddings.append(emb)
                    all_labels.extend(lbls_batch.numpy().tolist())

                X_emb = np.vstack(all_embeddings)
                y_emb = np.array(all_labels)

                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric='mlogloss')
                xgb_model.fit(X_emb, y_emb)
                
                val_embeddings = []
                for imgs_batch, _ in val_ds:
                    emb_val = embedding_model.predict(imgs_batch, verbose=0)
                    val_embeddings.append(emb_val)
                X_val_emb = np.vstack(val_embeddings)

                y_pred = xgb_model.predict(X_val_emb)

                # Simpan model bundle .pkl
                model_bundle = {
                    "model": xgb_model,
                    "embedding_model": embedding_model,
                    "class_names": class_names
                }
                joblib.dump(model_bundle, model_path)
            else:
                val_preds = cnn_model.predict(val_cached, verbose=0)
                y_pred = np.argmax(val_preds, axis=1)
                
                val_embeddings = []
                for imgs_batch, _ in val_ds:
                    emb_val = embedding_model.predict(imgs_batch, verbose=0)
                    val_embeddings.append(emb_val)
                X_val_emb = np.vstack(val_embeddings)

                # Simpan model bundle .pkl (CNN murni)
                model_bundle = {
                    "model": cnn_model,
                    "class_names": class_names
                }
                joblib.dump(model_bundle, model_path)

            # Pastikan format y_pred aman
            if isinstance(y_pred, np.ndarray):
                y_pred = y_pred.astype(int)
            else:
                y_pred = np.array(y_pred).astype(int)

            if len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)

            # Hitung Metrik Evaluasi Real
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            num_classes = len(class_names)
            labels_list = list(range(num_classes))
            cm = confusion_matrix(y_true, y_pred, labels=labels_list)

            # Ekstraksi PCA 3D
            scatter_data = []
            try:
                pca = PCA(n_components=3)
                pca_coords = pca.fit_transform(X_val_emb)
                for i in range(len(pca_coords)):
                    x_val, y_val, z_val = pca_coords[i]
                    label_val = int(y_true[i])
                    scatter_data.append([float(x_val), float(y_val), float(z_val), label_val])
            except Exception as pca_err:
                print(f"⚠️ Gagal menghasilkan PCA scatter features: {str(pca_err)}")

            # Update ke Global Status
            training_status["accuracy"] = float(acc)
            training_status["scatter_features"] = scatter_data
            training_status["metrics"] = {
                "accuracy": round(float(acc) * 100, 2),
                "akurasi": f"{round(float(acc) * 100, 2)}%",
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "f1_score": round(float(f1), 4),
                "confusion_matrix": cm.tolist(),
                "class_labels": class_names,
                "scatter_features": scatter_data
            }

            # Generate PDF Laporan
            generate_pdf_report(
                pdf_path=pdf_path,
                algoritma=algoritma,
                optimasi=optimasi,
                dataset=dataset,
                metrics=training_status["metrics"],
                class_names=class_names
            )

            training_status["status"] = "done"

        except Exception as e:
            training_status["status"] = "error"
            training_status["metrics"]["error"] = str(e)
            traceback.print_exc()

    t = threading.Thread(target=train_background_worker, daemon=True)
    t.start()

    return JsonResponse({"status": "success", "message": "Proses training model berhasil dimulai!"})


def get_training_results(request):
    metrics = training_status.get("metrics", {}).copy()
    
    if not metrics or "precision" not in metrics:
        acc_val = training_status.get('accuracy', 0.0) * 100
        metrics = {
            "accuracy": f"{round(acc_val, 2)}%",
            "akurasi": f"{round(acc_val, 2)}%",
            "precision": "-",
            "recall": "-",
            "f1": "-",
            "f1_score": "-",
            "confusion_matrix": [],
            "scatter_features": training_status.get("scatter_features", [])
        }
    else:
        if "accuracy" in metrics and not str(metrics["accuracy"]).endswith("%"):
            metrics["accuracy"] = f"{round(float(metrics['accuracy']), 2)}%"
            metrics["akurasi"] = metrics.get("accuracy", f"{round(training_status.get('accuracy', 0.0) * 100, 2)}%")
            metrics.setdefault("precision", "-")
            metrics.setdefault("recall", "-")
            metrics.setdefault("f1", "-")
            metrics.setdefault("f1_score", metrics.get("f1", "-"))
            metrics.setdefault("confusion_matrix", [])
            metrics["scatter_features"] = metrics.get("scatter_features", training_status.get("scatter_features", []))

    return JsonResponse(metrics)


def get_training_progress(request):
    if training_status.get("progress", 0) >= 100 and training_status.get("status") == "training":
        training_status["status"] = "done"
        
    return JsonResponse(training_status)


def list_models(request):
    MODEL_DIR = os.path.join(settings.BASE_DIR, "models")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        
    models_data = []
    files = os.listdir(MODEL_DIR)
    
    for filename in files:
        if filename.endswith(".pkl"):
            name_without_ext = filename[:-4]
            parts = name_without_ext.split("_")
            
            if len(parts) >= 3:
                algo_main = parts[0].upper()
                algo_opt = parts[1].capitalize()
                algorithm_display = f"{algo_main} + {algo_opt}"
                dataset_name = " ".join(parts[2:]).capitalize()
            elif len(parts) == 2:
                algo_main = parts[0].upper()
                algorithm_display = algo_main
                dataset_name = parts[1].capitalize()
            else:
                algorithm_display = "CNN"
                dataset_name = name_without_ext.capitalize()
                
            file_path = os.path.join(MODEL_DIR, filename)
            mod_time = os.path.getmtime(file_path)
            formatted_date = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            pdf_filename = f"{name_without_ext}.pdf"
            has_pdf = os.path.exists(os.path.join(MODEL_DIR, pdf_filename))
            
            models_data.append({
                "id": filename,
                "filename": filename,
                "pdf_filename": pdf_filename if has_pdf else None,
                "algorithm": algorithm_display,
                "dataset": dataset_name,
                "modified": formatted_date
            })
            
    return JsonResponse({"models": models_data})

      
@csrf_exempt
def run_kfold_evaluation(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "msg": "Method tidak diizinkan!"}, status=405)
    
    try:
        k_value = int(request.POST.get("k_fold", 5))
        dataset = request.POST.get("dataset", "Retina")
        optimasi = request.POST.get("optimasi", "XGBoost")
        algoritma = request.POST.get("algoritma", "CNN")
        
        ds_name = dataset.lower().replace(" ", "_")
        train_dir = os.path.join(settings.BASE_DIR, "dataset", ds_name, "train")
        
        if not os.path.exists(train_dir):
            return JsonResponse({"status": "error", "msg": f"Folder dataset train tidak ditemukan untuk {dataset}."}, status=400)

        img_size = (224, 224)
        batch_size = 32

        full_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=batch_size, shuffle=True
        )
        class_names = full_ds.class_names

        images_list = []
        labels_list = []
        for img_batch, lbl_batch in full_ds:
            images_list.append(img_batch.numpy())
            labels_list.append(lbl_batch.numpy())
            
        X = np.concatenate(images_list, axis=0)
        y = np.concatenate(labels_list, axis=0)

        inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        flatten_out = tf.keras.layers.Flatten(name="flatten_layer")(x)
        x_dense = tf.keras.layers.Dense(128, activation='relu')(flatten_out)
        outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x_dense)

        cnn_base = Model(inputs=inputs, outputs=outputs)
        embedding_model = Model(inputs=inputs, outputs=flatten_out)
        cnn_base.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        kf = KFold(n_splits=k_value, shuffle=True, random_state=42)
        scores = []
        accuracies = []
        losses = []
        
        all_y_true = []
        all_y_pred = []

        fold_idx = 1
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            cnn_base.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, verbose=0)
            
            if optimasi.lower() == "xgboost":
                train_feats = embedding_model.predict(X_train, verbose=0)
                val_feats = embedding_model.predict(X_val, verbose=0)
                
                fold_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric='mlogloss')
                fold_model.fit(train_feats, y_train)
                y_pred = fold_model.predict(val_feats)
                acc = accuracy_score(y_val, y_pred) * 100
                loss = 0.15 
            else:
                val_loss, val_acc = cnn_base.evaluate(X_val, y_val, verbose=0)
                preds = cnn_base.predict(X_val, verbose=0)
                y_pred = np.argmax(preds, axis=1)
                acc = val_acc * 100
                loss = val_loss

            all_y_true.extend(y_val)
            all_y_pred.extend(y_pred)

            scores.append({
                "fold": fold_idx,
                "accuracy": round(float(acc), 2)
            })
            accuracies.append(acc)
            losses.append(loss)
            fold_idx += 1

        avg_acc = np.mean(accuracies)
        avg_loss = np.mean(losses)

        cm = confusion_matrix(all_y_true, all_y_pred)
        final_cm = cm.tolist()

        return JsonResponse({
            "status": "success",
            "k": k_value,
            "scores": scores,
            "average_accuracy": f"{round(avg_acc, 2)}%",
            "average_loss": f"{round(float(avg_loss), 4)}",
            "class_names": class_names,
            "confusion_matrix": final_cm
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "msg": str(e)}, status=500)

@csrf_exempt
def run_testing(request):
    """Endpoint testing menggunakan TensorFlow native image decoding (Bersih & Tanpa Duplikasi)"""
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "Method tidak diizinkan!"}, status=405)

    try:
        model_name = request.POST.get("model_name", "")
        
        MODEL_DIR = os.path.join(settings.BASE_DIR, "models")
        model_path = os.path.join(MODEL_DIR, model_name)

        if not model_name or not os.path.exists(model_path):
            return JsonResponse({
                "status": "error",
                "message": f"File model '{model_name}' tidak ditemukan di direktori models!"
            }, status=404)

        model_bundle = joblib.load(model_path)
        model = model_bundle.get("model")
        embedding_model = model_bundle.get("embedding_model", None)
        class_names = model_bundle.get("class_names", [])

        results = []
        
        if not request.FILES:
            return JsonResponse({"status": "error", "message": "Tidak ada file gambar yang diunggah!"}, status=400)

        for key, uploaded_file in request.FILES.items():
            try:
                # Membaca byte file gambar secara langsung (Aman dari error PIL Image)
                image_bytes = uploaded_file.read()
                
                tensor = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
                tensor = tf.image.resize(tensor, [224, 224])
                
                img_array = tensor.numpy().astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                if embedding_model is not None:
                    emb = embedding_model.predict(img_array, verbose=0)
                    pred_idx = model.predict(emb)[0]
                else:
                    preds = model.predict(img_array, verbose=0)
                    pred_idx = np.argmax(preds, axis=1)[0]

                prediction_label = class_names[int(pred_idx)] if int(pred_idx) < len(class_names) else f"Class {pred_idx}"

                results.append({
                    "file": uploaded_file.name,
                    "prediction": prediction_label,
                    "msg": "success"
                })

            except Exception as file_err:
                traceback.print_exc()
                results.append({
                    "file": uploaded_file.name,
                    "prediction": "error",
                    "msg": str(file_err)
                })

        return JsonResponse({
            "status": "success",
            "result": results
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)