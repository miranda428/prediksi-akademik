RINGKASAN COLAB + STREAMLIT
===========================

COLAB 01
01_Pelatihan_Model_Naive_Bayes_Non_Akademik.ipynb
= proses penelitian/modeling:
dataset -> target -> preprocessing -> split 80:20 -> Gaussian Naive Bayes -> evaluasi -> joblib.

COLAB 02
02_Jalankan_Streamlit_Non_Akademik_di_Colab.ipynb
= menjalankan aplikasi web yang sudah jadi. Tidak melatih model.
Notebook ini memakai Cloudflare Quick Tunnel untuk URL sementara.

STREAMLIT ONLINE
= mengambil aplikasi.py dari GitHub dan menjalankannya di Streamlit Community Cloud.
Main file: aplikasi.py

DATABASE ONLINE
= jika DATABASE_URL tersedia di Streamlit Secrets, aplikasi memakai PostgreSQL/Supabase.
Tanpa DATABASE_URL, aplikasi lokal memakai SQLite.

KEAMANAN
Jangan commit .streamlit/secrets.toml atau password database ke GitHub.
