============================================================
SISTEM PREDIKSI NON AKADEMIK — PAKET FINAL
Implementasi Algoritma Naive Bayes untuk Prediksi
Prestasi Non Akademik Siswa Berbasis Web
============================================================

A. FILE UTAMA

1. aplikasi.py
   Kode sumber utama aplikasi Streamlit. File ini memuat model,
   preprocessing, login, prediksi, data siswa, perkembangan,
   konsultasi, evaluasi model, dan koneksi database.

2. gaussian_naive_bayes.joblib
   Model Gaussian Naive Bayes yang sudah dilatih dan disimpan
   menggunakan joblib.

3. preprocessor.joblib
   Pipeline preprocessing yang digunakan saat training. File ini
   memastikan data input aplikasi diproses konsisten dengan data latih.

4. metadata.json
   Informasi pendukung model: 19 prediktor, pembagian data, target,
   parameter model, metrik, dan matriks konfusi.

5. StudentPerformanceFactors.csv
   Dataset sekunder yang digunakan untuk pembentukan dan pengujian model.

6. requirements.txt
   Dependensi Python yang dipakai aplikasi. Versi dipin untuk menjaga
   kompatibilitas model joblib dan lingkungan deployment.

7. PANDUAN_DATABASE_ONLINE.txt
   Petunjuk koneksi PostgreSQL/Supabase melalui Streamlit Secrets.

8. README_COLAB_STREAMLIT.txt
   Ringkasan penggunaan Colab dan deployment.

9. .gitignore
   Mencegah file rahasia dan database lokal ikut masuk GitHub.

B. STRUKTUR PROSES PENELITIAN

Dataset
  -> preprocessing
  -> pembagian data 80% data latih / 20% data uji
  -> Gaussian Naive Bayes
  -> evaluasi
  -> model .joblib
  -> aplikasi Streamlit

Model final yang diverifikasi:
- Dataset: 6.607 baris
- Prediktor: 19
- Data latih: 5.285
- Data uji: 1.322
- Random state: 42
- var_smoothing: 0.000308
- Accuracy: 84,80%
- Precision weighted: 85,76%
- Recall weighted: 84,80%
- F1-score weighted: 85,04%

C. GOOGLE COLAB

Gunakan dua notebook:

01_Pelatihan_Model_Naive_Bayes_Non_Akademik.ipynb
- menjelaskan dataset
- membentuk target Rendah/Sedang/Tinggi
- preprocessing
- split 80:20
- training Gaussian Naive Bayes
- evaluasi
- membuat model.joblib, preprocessor.joblib, metadata.json

02_Jalankan_Streamlit_Non_Akademik_di_Colab.ipynb
- menjalankan aplikasi.py
- mengecek port 8501
- menggunakan Cloudflare Quick Tunnel untuk URL sementara
- tidak menggunakan LocalTunnel

D. MENJALANKAN LOKAL

1. Buka terminal pada folder aplikasi.
2. Jalankan:
   pip install -r requirements.txt
3. Jalankan:
   streamlit run aplikasi.py

E. GITHUB

Untuk repository GitHub, upload ISI folder ini ke root repository.
File utama harus berada sejajar dengan requirements.txt:

aplikasi.py
requirements.txt
gaussian_naive_bayes.joblib
preprocessor.joblib
metadata.json
StudentPerformanceFactors.csv

README dan file panduan boleh ikut di-upload.

Jangan upload:
- .streamlit/secrets.toml
- sistem.db
- password database
- file rahasia lainnya

F. STREAMLIT COMMUNITY CLOUD

Entrypoint/Main file:
  aplikasi.py

Branch:
  main

Pastikan requirements.txt berada di root repository atau di folder yang
sama dengan entrypoint. Community Cloud membaca dependency dari file tersebut.

G. SUPABASE / POSTGRESQL ONLINE

Saat DATABASE_URL tersedia di Streamlit Secrets, aplikasi otomatis
menggunakan PostgreSQL. Jika DATABASE_URL tidak tersedia, aplikasi lokal
menggunakan SQLite (sistem.db).

Di Streamlit Secrets, simpan:

DATABASE_URL = "postgresql://USER:PASSWORD@HOST:PORT/postgres"

Ambil URI langsung dari tombol Connect di Supabase. Untuk shared transaction
pooler, gunakan URI yang disediakan Supabase (umumnya port 6543) dan jangan
menebak host/username. Jika password mengandung karakter khusus, ikuti format
encoding yang disediakan Supabase.

Tabel dibuat otomatis saat koneksi pertama:
- users
- students
- predictions
- consultations
- development

H. FITUR YANG SUDAH DICEK

- Login
- Dashboard
- Prediksi Prestasi
- Data Siswa
- Perkembangan
- Evaluasi Model
- Matriks Konfusi
- Konsultasi kirim pesan
- Konsultasi balasan
- SQLite lokal
- PostgreSQL/Supabase untuk mode online

I. CATATAN KEAMANAN

Password akun demo pada source code hanya untuk pengujian/skripsi.
Untuk penggunaan operasional nyata, ganti mekanisme akun dan kredensial.
Jangan pernah memasukkan password Supabase ke source code atau GitHub.

J. CATATAN PENTING UNTUK SKRIPSI

Angka evaluasi resmi harus konsisten dengan model final yang telah
diverifikasi. Jika model dilatih ulang, jangan langsung mengganti model
final skripsi sebelum hasil, metadata, Bab III, dan Bab IV diperiksa kembali.
