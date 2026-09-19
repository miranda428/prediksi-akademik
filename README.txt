SISTEM PREDIKSI NON AKADEMIK - NAIVE BAYES
===========================================

Isi paket:
- aplikasi.py                 : versi Non Akademik utama
- app.py                      : salinan yang sama, untuk entrypoint Streamlit jika diperlukan
- aplikasi_lama_backup.py     : backup aplikasi versi lama
- model/gaussian_naive_bayes.joblib
- model/preprocessor.joblib
- model/metadata.json
- data/StudentPerformanceFactors.csv
- requirements.txt
- PANDUAN_DATABASE_ONLINE.txt
- .gitignore

CARA PINDAH KE GITHUB
1. Upload/replace aplikasi.py pada repository lama.
2. Upload folder model dan isinya.
3. Upload folder data dan isinya.
4. Pastikan requirements.txt ikut.
5. Jangan menghapus konfigurasi Secrets DATABASE_URL di Streamlit.
6. Jika Streamlit menggunakan aplikasi.py sebagai Main file, biarkan aplikasi.py.
   Jika menggunakan app.py, file app.py sudah disediakan dengan isi yang sama.

CATATAN DATABASE
Aplikasi mendukung SQLite lokal dan PostgreSQL/Supabase melalui DATABASE_URL.
Secrets DATABASE_URL tidak disertakan dalam paket demi keamanan.
Jangan memasukkan password Supabase ke GitHub.
