import threading, time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ml_lib import cnn_model, c45_model, kmeans_model, optimizers

training_progress = {"status": "idle", "percent": 0, "message": ""}
training_results = {}

def update_progress(p, msg=""):
    training_progress["percent"] = int(p)
    training_progress["message"] = msg
    training_progress["status"] = "running" if p < 100 else "done"

@csrf_exempt
def start_training(request):
    algoritma = request.POST.get("algoritma", "").lower()
    kategori = request.POST.get("kategori", "").lower()
    optimasi = request.POST.get("optimasi", "").lower()
    dataset = request.POST.get("dataset", "").lower()

    if not all([algoritma, kategori, dataset]):
        return JsonResponse({"error": "Parameter tidak lengkap."}, status=400)

    def training_thread():
        try:
            update_progress(0, f"Menyiapkan dataset '{dataset}' untuk {algoritma.upper()}...")
            time.sleep(1)

            # Pilih algoritma utama
            if algoritma == "cnn":
                results = cnn_model.train_cnn(dataset, update_progress)
            elif algoritma == "c45":
                results = c45_model.train_c45(dataset, update_progress)
            elif algoritma == "kmeans":
                results = kmeans_model.train_kmeans(dataset, update_progress)
            else:
                update_progress(100, f"❌ Algoritma '{algoritma}' belum didukung.")
                return

            # Terapkan optimasi
            if optimasi not in ["", "none", "-", "tidak ada"]:
                update_progress(80, f"Menerapkan optimasi {optimasi.upper()}...")
                results = optimizers.apply_optimizer(results, optimasi, update_progress)

            # Update hasil akhir
            training_results.clear()
            training_results.update(results)
            update_progress(100, f"Training {algoritma.upper()} selesai!")

        except Exception as e:
            update_progress(100, f"❌ Terjadi error: {str(e)}")

    threading.Thread(target=training_thread).start()
    return JsonResponse({"status": "started", "message": f"Training {algoritma.upper()} dimulai..."})

def get_progress(request):
    return JsonResponse(training_progress)

def get_results(request):
    return JsonResponse(training_results)
