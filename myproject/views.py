import os, datetime
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import joblib
import tensorflow as tf
from difflib import get_close_matches

# import custom libraries
from ml_lib import cnn_model, c45_model, xgboost_model, adaboost_model, pso_optimizer, kmeans_model, kflod_cv, plot_utils

DATA_DIR = os.path.join(settings.BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


# Progress tracking global
training_state = {
    "progress": 0,
    "status": "idle",
    "detail": "",
    "metrics": None
}



# === Fungsi utilitas ===
def model_path(filename):
    model_dir = os.path.join(settings.BASE_DIR, "dashboard", "models")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, filename)


def list_models(request):
    possible_dirs = [
        os.path.join(settings.BASE_DIR, "dashboard", "models"),
        os.path.join(settings.BASE_DIR, "models"),
    ]

    found_files = []
    for model_dir in possible_dirs:
        if not os.path.exists(model_dir):
            continue

        for f in os.listdir(model_dir):
            if f.endswith(".pkl"):
                full_path = os.path.join(model_dir, f)
                modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
                found_files.append({
                    "filename": f,
                    "path": model_dir,
                    "algorithm": f.split("_")[0] if "_" in f else "Unknown",
                    "dataset": f.split("_")[-1].replace(".pkl", "") if "_" in f else "Unknown",
                    "size_kb": round(os.path.getsize(full_path) / 1024, 2),
                    "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S")
                })

    found_files.sort(key=lambda x: x["modified"], reverse=True)

    return JsonResponse({
        "status": "success",
        "count": len(found_files),
        "models": found_files
    })


# ------------------------
# Utility functions
# ------------------------
def load_dataset(name):
    if name == 'car-damage':
        # dummy tabular dataset
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
    elif name == 'fetus':
        X = np.random.rand(100, 5)
        y = None
    elif name == 'retina':
        X = np.random.rand(100, 64, 64, 3)
        y = np.random.randint(0, 4, 100)
    else:
        X = y = None
    return X, y

def load_model(alg, ds_name, opt=None):
    """
    Load model untuk training atau testing.
    
    Args:
        alg (str): algoritma utama (CNN, C45, KMeans, dll)
        ds_name (str): nama dataset
        opt (str, optional): optimasi (XGBoost, Adaboost, PSO, dll)

    Returns:
        model object atau None jika tidak ditemukan
    """
    # Tentukan nama file berdasarkan parameter
    if opt:
        filename = f'{alg}_{opt}_{ds_name}.model'
    else:
        filename = f'{alg}_{ds_name}.model'
    
    path = os.path.join(MODEL_DIR, filename)

    # Load model sesuai jenis algoritma
    if alg == 'CNN' and os.path.exists(path):
        return tf.keras.models.load_model(path)
    elif alg in ['C45', 'XGBoost', 'AdaBoost', 'KMeans'] and os.path.exists(path):
        return joblib.load(path)
    else:
        return None

def load_model(alg, ds_name, opt=None):
    """
    Load model untuk training atau testing.
    """
    filename = f'{alg}_{opt}_{ds_name}.model' if opt else f'{alg}_{ds_name}.model'
    path = os.path.join(MODEL_DIR, filename)

    if alg == 'CNN' and os.path.exists(path):
        return tf.keras.models.load_model(path)
    elif alg in ['C45', 'XGBoost', 'AdaBoost', 'KMeans'] and os.path.exists(path):
        return joblib.load(path)
    else:
        return None
# ------------------------
# Views
# ------------------------
def index(request):
    return render(request, 'myproject/index.html')

def dashboard(request):
    return render(request, 'myproject/dashboard.html')


@csrf_exempt
def start_training(request):
    if request.method == 'POST':
        alg = request.POST.get('algoritma')
        ds_name = request.POST.get('dataset')
        optim = request.POST.get('optimasi')
        X, y = load_dataset(ds_name)
        model = None

        if alg == 'CNN':
            model = cnn.train(X, y)
            model.save(os.path.join(MODEL_DIR, f'{alg}_{ds_name}.model'))
        elif alg == 'C45':
            model = c45.train(X, y)
            joblib.dump(model, os.path.join(MODEL_DIR, f'{alg}_{ds_name}.model'))
        elif alg == 'KMeans':
            model = kmeans_lib.train(X, n_clusters=3)
            joblib.dump(model, os.path.join(MODEL_DIR, f'{alg}_{ds_name}.model'))

        return JsonResponse({'status':'success', 'msg':'Training selesai'})


def load_model_testing(alg, opt, ds_name):
    """
    Load model .pkl yang paling mirip dengan input form.
    """
    desired_name = f"{alg}_{opt}_{ds_name}.pkl"
    all_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    # Cari yang paling mirip
    matches = get_close_matches(desired_name, all_models, n=1, cutoff=0.1)
    if matches:
        path = os.path.join(MODEL_DIR, matches[0])
        return joblib.load(path)
    
    # Jika tidak ada yang mirip, return None
    return None

def preprocess_image(file, ds_name):
    """
    Load image, resize sesuai dataset, ubah jadi array.
    """
    img = Image.open(file).convert('RGB')
    if ds_name == 'retina':
        img = img.resize((64,64))
        arr = np.array(img).reshape(1,64,64,3) / 255.0
    else:
        # default resize misal 32x32
        img = img.resize((32,32))
        arr = np.array(img).reshape(1,32*32*3) / 255.0
    return arr

@csrf_exempt
def run_testing(request):
    if request.method == 'POST':
        alg = request.POST.get('algoritma')
        opt = request.POST.get('optimasi')
        ds_name = request.POST.get('dataset')
        selectModel = request.POST.get('select-model')
        files = request.FILES.getlist('test_file')
        results = []

        model = load_model_testing(alg, opt, ds_name)
        if model is None:
            return JsonResponse({
                'status':'error',
                'msg':'Model belum ada',
                'expected_path': os.path.join(MODEL_DIR, f'{alg}_{opt}_{ds_name}.pkl')
            })

        for f in files:
            try:
                X_test = preprocess_image(f, ds_name)

                # Prediksi sesuai algoritma
                if alg in ['C45', 'XGBoost', 'AdaBoost', 'KMeans']:
                    pred = model.predict(X_test)
                    pred_value = int(pred[0])
                else:
                    pred_value = 'unknown'

                results.append({'file': f.name, 'prediction': pred_value})
            except Exception as e:
                results.append({'file': f.name, 'prediction': 'error', 'msg': str(e)})

        return JsonResponse({'status': 'success', 'result': results})
    else:
        return JsonResponse({'status': 'error', 'msg': 'Method harus POST'})


@csrf_exempt
def run_evaluation(request):
    if request.method == 'POST':
        kfold = int(request.POST.get('kfold', 5))
        alg = request.POST.get('algoritma')
        ds_name = request.POST.get('dataset')

        X, y = load_dataset(ds_name)
        model = load_model(alg, ds_name)
        if model is None:
            return JsonResponse({'status':'error','msg':'Model belum ada'})

        if alg in ['C45', 'XGBoost', 'AdaBoost']:
            res = kfold_cv.evaluate(model, X, y, k=kfold)
        else:
            res = {'msg':'K-Fold hanya untuk model tabular/sklearn'}

        return JsonResponse({'status':'success','kfold_results':res})
