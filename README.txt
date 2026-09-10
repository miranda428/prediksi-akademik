==================================================
AkademikPredict
Implementasi Algoritma Naive Bayes untuk Prediksi
Prestasi Akademik Siswa Berbasis Web
==================================================

Sistem prediksi kategori prestasi akademik siswa berbasis web,
menggunakan algoritma Gaussian Naive Bayes (GNB).

Prototipe ini dibangun untuk keperluan penelitian skripsi dan
demonstrasi sidang.

==================================================
1. CARA INSTALASI
==================================================

Prasyarat: Python 3.10 atau lebih baru.

1. (Opsional tapi disarankan) Buat virtual environment:

   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows

2. Install seluruh dependensi sesuai versi yang sudah dipin
   (versi ini konsisten dengan model yang sudah dilatih —
   jangan mengganti versi scikit-learn secara sembarangan,
   karena dapat memicu ketidaksesuaian saat memuat model):

   pip install -r requirements.txt

==================================================
2. CARA MENJALANKAN APLIKASI
==================================================

Dari folder utama proyek, jalankan:

   streamlit run app.py

Aplikasi akan terbuka otomatis di browser pada:

   http://localhost:8501

Database operasional SQLite (sistem.db) akan dibuat otomatis
pada folder yang sama saat aplikasi pertama kali dijalankan,
lengkap dengan akun demo dan 10 data siswa awal (4 Rendah, 3 Sedang, 3 Tinggi).
Data tambah/edit/hapus siswa langsung di-commit ke database.

==================================================
3. AKUN DEMO
==================================================

| Peran              | Username    | Password       |
|--------------------|-------------|----------------|
| Super Admin        | admin       | Admin@123      |
| Operator           | operator01  | Operator@123   |
| Guru/Wali Kelas    | GURU001     | Guru@123       |
| Kepala Sekolah     | KEPSEK01    | Kepsek@123     |
| Siswa              | NIS001–NIS010 | Siswa@123    |
| Orang Tua/Wali     | ORTU001       | Ortu@123     |

Catatan keamanan: akun-akun di atas hanya untuk pengujian/demo.
Ganti seluruh password dan lakukan pengamanan tambahan (HTTPS,
database terkelola, dsb.) sebelum digunakan pada lingkungan nyata.

Hak akses per peran (least privilege):
- Siswa hanya dapat melihat & menjalankan prediksi untuk dirinya
  sendiri, serta hanya melihat riwayat/perkembangan miliknya sendiri.
- Orang Tua/Wali hanya dapat melihat riwayat/perkembangan siswa
  yang terhubung dengan akunnya.
- Guru/Wali Kelas, Operator, dan Super Admin memiliki akses lebih
  luas sesuai kewenangan masing-masing pada sistem.

==================================================
4. STRUKTUR FOLDER
==================================================

Sistem_AkademikPredict_GNB_FINAL/
├── app.py                          Aplikasi utama (Streamlit, single-file)
├── requirements.txt                Daftar dependensi Python (versi dipin)
├── README.txt                      Dokumen ini
├── (folder model/ tidak diperlukan pada versi GitHub Ready) 
│   ├── gaussian_naive_bayes.joblib Model Gaussian Naive Bayes terlatih
│   ├── preprocessor.joblib         Pipeline praproses (imputasi + one-hot encoding)
│   └── metadata.json               Metadata dataset, parameter model, dan hasil evaluasi
└── (folder data/ tidak diperlukan pada versi GitHub Ready) 
    └── StudentPerformanceFactors.csv   Dataset penelitian (data sekunder)

Catatan: file sistem.db (database operasional SQLite) TIDAK disertakan
dalam paket ini karena dibuat otomatis saat aplikasi pertama kali
dijalankan. Pada penggunaan lokal, sistem.db tetap menyimpan perubahan
meskipun browser ditutup. Menu Data Siswa menyediakan Tambah, Edit,
dan Hapus; setiap aksi langsung disimpan dan halaman dimuat ulang.

==================================================
5. TEKNOLOGI
==================================================

- Python
- Streamlit        — antarmuka web
- Pandas, NumPy    — pengolahan data
- Scikit-learn     — model Gaussian Naive Bayes & pipeline praproses
- Joblib           — penyimpanan/pemuatan model
- SQLite           — basis data operasional (akun, data siswa,
                      riwayat prediksi, catatan perkembangan, konsultasi)

Metodologi data mining: CRISP-DM.

==================================================
6. SUMBER DATASET (DATA SEKUNDER)
==================================================

Dataset penelitian merupakan data sekunder "Student Performance
Factors" yang diperoleh dari Kaggle, dan BUKAN merupakan data siswa
SMP Global Persada Mandiri.

Dataset digunakan untuk membangun dan menguji model prediksi,
sedangkan SMP Global Persada Mandiri merupakan lokasi pelaksanaan
penelitian dan uji coba sistem (bukan sumber data pelatihan model).

Ringkasan dataset:
- 6.607 baris data
- 20 kolom (19 variabel prediktor + 1 target: Exam_Score)
- Tidak ada duplicate rows
- Missing value ditemukan pada 3 kolom (Teacher_Quality,
  Parental_Education_Level, Distance_from_Home), ditangani dengan
  imputasi nilai modus penelitian (lihat metadata.json untuk detail)

Kategori target (aturan penelitian berdasarkan distribusi dataset,
bukan standar penilaian resmi sekolah):
- Rendah : Exam_Score <= 66   (2.882 data)
- Sedang : Exam_Score 67–69   (2.100 data)
- Tinggi : Exam_Score >= 70   (1.625 data)

==================================================
7. MODEL GAUSSIAN NAIVE BAYES
==================================================

- Algoritma  : Gaussian Naive Bayes (GNB)
- Parameter  : var_smoothing = 0.000308
- Preprocessing:
    * Variabel numerik  -> imputasi median, TANPA standardisasi
    * Variabel kategorikal -> imputasi modus + one-hot encoding
    * Tidak ada feature selection (seluruh 19 variabel digunakan)
- Pembagian data: stratified train-test split 80:20,
  random_state = 42
    * Data latih : 5.285 baris
    * Data uji   : 1.322 baris

19 variabel prediktor:
Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources,
Extracurricular_Activities, Sleep_Hours, Previous_Scores,
Motivation_Level, Internet_Access, Tutoring_Sessions, Family_Income,
Teacher_Quality, School_Type, Peer_Influence, Physical_Activity,
Learning_Disabilities, Parental_Education_Level, Distance_from_Home,
Gender

==================================================
8. HASIL EVALUASI MODEL (FINAL — TIDAK BOLEH DIUBAH)
==================================================

Accuracy           : 84.80%
Weighted Precision  : 85.76%
Weighted Recall     : 84.80%
Weighted F1-Score   : 85.04%

Confusion Matrix (baris = aktual, kolom = prediksi;
urutan kelas: Rendah, Sedang, Tinggi):

              Prediksi Rendah   Prediksi Sedang   Prediksi Tinggi
Aktual Rendah        498               79                0
Aktual Sedang         38              356               26
Aktual Tinggi          5               53              267

Hasil evaluasi di atas diperoleh dari data uji (20%) dengan
pemisahan berstrata, dan telah diverifikasi ulang secara independen
langsung dari dataset mentah + model tersimpan — angka bersifat
final dan konsisten dengan laporan skripsi.

==================================================
CATATAN PENUTUP
==================================================

Sistem ini adalah prototipe akademik untuk keperluan penelitian dan
demonstrasi sidang skripsi. Untuk penggunaan produksi/nyata, lakukan
langkah tambahan seperti: mengganti seluruh akun & password demo,
menggunakan HTTPS, migrasi ke database terkelola bila diperlukan,
dan audit keamanan lebih lanjut.
