Berikut adalah file **`README.md`** lengkap, profesional, dan siap pakai untuk proyek Django CNN + XGBoost Anda:

```markdown
# Django CNN + XGBoost Image Processing

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.x-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## Deskripsi

Aplikasi **Django** full-stack untuk klasifikasi gambar menggunakan arsitektur hibrida **Convolutional Neural Network (CNN)** sebagai ekstraktor fitur (*feature extractor*) dan **XGBoost** sebagai pengklasifikasi akhir (*classifier*). Aplikasi ini dirancang untuk memudahkan proses pelatihan model secara asinkron, evaluasi metrik mendalam, hingga pengujian gambar secara *real-time*.

### Fitur Utama:
- **Form Pilihan Dinamis**: Antarmuka berbasis kartu (*card selection*) untuk memilih kombinasi algoritma, kategori tujuan, jenis optimasi, dan dataset.
- **Background Training**: Melatih model CNN + XGBoost di latar belakang menggunakan *threading* dengan pelacakan *progress bar* secara *real-time*.
- **Evaluasi Model Komprehensif**: Menampilkan metrik Akurasi, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix* multi-kelas.
- **Pengujian Multi-File (Testing)**: Unggah beberapa gambar sekaligus dengan penanganan channel otomatis (Grayscale/RGBA ke RGB) dan pratinjau instan.
- **Manajemen Model**: Menyimpan dan memuat kembali model hibrida menggunakan `joblib` serta mengelolanya dalam tabel daftar model aktif.

---

## Struktur Project

```text
project_root/
├── dataset/                  # Folder dataset (train/val per kelas)
├── models/                   # Direktori penyimpanan file model (.pkl)
├── classifier/               # Aplikasi utama Django
│   ├── templates/            # Template HTML dashboard & modal
│   ├── views.py              # Logika training, testing, dan evaluasi
│   ├── urls.py               # Konfigurasi routing URL
│   └── ...
├── media/                    # File unggahan pengguna
├── static/                   # Aset statis (CSS/JS)
├── db.sqlite3                # Database SQLite
├── manage.py                 # Utilitas baris perintah Django
└── README.md

```

---

## Persyaratan Sistem

* **Python** 3.11+
* **Django** 4.x
* **TensorFlow** 2.x
* **XGBoost** & **scikit-learn**
* **Joblib**, **NumPy**, **Pillow**, & **Plotly** (untuk visualisasi fitur)

---

## Instalasi & Konfigurasi

1. **Clone repository & masuk ke direktori project:**
```bash
cd project_root

```


2. **Aktifkan Virtual Environment:**
* **Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1

```


* **Linux / macOS:**
```bash
source .venv/bin/activate

```




3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Migrasi Database:**
```bash
python manage.py makemigrations
python manage.py migrate

```



---

## Cara Menjalankan Aplikasi

1. **Jalankan Server Django:**
```bash
python manage.py runserver

```


2. **Akses Aplikasi melalui Browser:**
Buka tautan berikut di browser Anda:
`http://127.0.0.1:8000/`
3. **(Opsional) Jalankan Celery Worker:**
Jika Anda menggunakan antrean tugas latar belakang berbasis Celery:
```bash
celery -A myproject worker -l info

```



---

## Panduan Pengujian (Testing)

Untuk menjalankan rangkaian unit test pada Django, gunakan perintah berikut:

```bash
# Menjalankan seluruh test
python manage.py test

# Menjalankan test spesifik pada modul tertentu
python manage.py test tests
python manage.py test tests.test_unit

```

---

## Alur Penggunaan Fitur

1. **Form Pilihan**: Pilih Algoritma (`CNN`), Kategori (`Klasifikasi`), Optimasi (`XGBoost`), dan Dataset yang diinginkan.
2. **Training**: Klik tombol *Start Training*. Pantau progress pelatihan secara langsung. Setelah selesai, klik *Lihat Hasil* untuk memunculkan modal metrik evaluasi dan *Confusion Matrix*.
3. **Daftar Model**: Model yang berhasil dilatih akan otomatis masuk ke dalam tabel daftar model. Pilih model yang ingin diuji menggunakan tombol radio.
4. **Testing**: Unggah file gambar pada bagian testing, lalu klik *Run Test* untuk melihat hasil prediksi label kelas secara instan.

---

## Lisensi

MIT License © 2026

```

```