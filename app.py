import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sqlite3, hashlib, secrets
from datetime import datetime

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Prediksi Akademik | Prediksi Prestasi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(__file__)
MD = BASE
DB = os.path.join(BASE, "sistem.db")

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(MD, "gaussian_naive_bayes.joblib"))
    pre = joblib.load(os.path.join(MD, "preprocessor.joblib"))
    with open(os.path.join(MD, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model, pre, meta

model, pre, meta = load_assets()

ROLES = {
    "Super Admin": ["Dashboard", "Prediksi Prestasi", "Data Siswa", "Perkembangan", "Konsultasi", "Evaluasi Model", "Data Penelitian", "Daftar Pengguna", "Panduan"],
    "Operator": ["Dashboard", "Prediksi Prestasi", "Data Siswa", "Perkembangan", "Konsultasi", "Evaluasi Model", "Data Penelitian", "Panduan"],
    "Guru/Wali Kelas": ["Dashboard", "Prediksi Prestasi", "Data Siswa", "Perkembangan", "Konsultasi", "Evaluasi Model", "Panduan"],
    "Kepala Sekolah": ["Dashboard", "Perkembangan", "Evaluasi Model", "Data Penelitian", "Panduan"],
    "Siswa": ["Dashboard", "Prediksi Prestasi", "Perkembangan", "Konsultasi", "Panduan"],
    "Orang Tua/Wali": ["Dashboard", "Perkembangan", "Konsultasi", "Panduan"],
}

DEFAULT_USERS = [
    ("admin", "Admin@123", "Administrator", "Administrator Sistem"),
    ("operator01", "Operator@123", "Operator", "Operator Akademik"),
    ("GURU001", "Guru@123", "Guru/Wali Kelas", "Guru/Wali Kelas"),
    ("KEPSEK01", "Kepsek@123", "Kepala Sekolah", "Kepala Sekolah"),
    ("NIS001", "Siswa@123", "Siswa", "Alya Putri"),
    ("NIS002", "Siswa@123", "Siswa", "Bagas Pratama"),
    ("NIS003", "Siswa@123", "Siswa", "Citra Lestari"),
    ("NIS004", "Siswa@123", "Siswa", "Daffa Ramadhan"),
    ("NIS005", "Siswa@123", "Siswa", "Eka Safitri"),
    ("NIS006", "Siswa@123", "Siswa", "Fajar Maulana"),
    ("NIS007", "Siswa@123", "Siswa", "Gita Maharani"),
    ("NIS008", "Siswa@123", "Siswa", "Hafiz Akbar"),
    ("NIS009", "Siswa@123", "Siswa", "Intan Permata"),
    ("NIS010", "Siswa@123", "Siswa", "Joko Saputra"),
    ("ORTU001", "Ortu@123", "Orang Tua/Wali", "Orang Tua/Wali"),
]

# 10 data awal untuk penggunaan awal sistem: 4 Rendah, 3 Sedang, 3 Tinggi.
# Nilai prediksi dibuat dari contoh baris dataset penelitian yang sudah diuji oleh model.
SEED_STUDENTS = [
    ("NIS001", "Alya Putri", "VIII A", "ORTU001", "Rendah", 0.9524505581, 1),
    ("NIS002", "Bagas Pratama", "VIII A", "", "Rendah", 0.5490648984, 7),
    ("NIS003", "Citra Lestari", "VIII A", "", "Rendah", 0.4803119101, 14),
    ("NIS004", "Daffa Ramadhan", "VIII B", "", "Rendah", 0.7348985297, 15),
    ("NIS005", "Eka Safitri", "VIII B", "", "Sedang", 0.4524689115, 0),
    ("NIS006", "Fajar Maulana", "VIII B", "", "Sedang", 0.5363657293, 6),
    ("NIS007", "Gita Maharani", "VIII B", "", "Sedang", 0.4808979676, 10),
    ("NIS008", "Hafiz Akbar", "VIII C", "", "Tinggi", 0.8240849805, 2),
    ("NIS009", "Intan Permata", "VIII C", "", "Tinggi", 0.4855818139, 3),
    ("NIS010", "Joko Saputra", "VIII C", "", "Tinggi", 0.6135918447, 4),
]

# ============================================================
# STYLE — putih + biru + lavender
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');
:root{--blue950:#10235f;--blue900:#17358f;--blue800:#2346a8;--blue700:#3159c9;--blue500:#4f83e8;--sky:#55b8f3;--cyan:#22c7d6;--violet:#8264df;--ink:#1d2433;--muted:#6f7d96;--line:#e4e9f3;--soft:#f5f7fc;--success:#16a56a;--warning:#e5a11a;--danger:#e05252;--shadow:0 10px 30px rgba(31,55,120,.09);--shadowHover:0 16px 36px rgba(31,55,120,.14)}
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stMarkdown,h1,h2,h3,h4{font-family:'Inter',sans-serif}.block-container{max-width:1500px;padding:1.2rem 2rem 3rem}[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f8faff,#f4f7fc)}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f2b73 0%,#163a91 48%,#2859bb 100%);border-right:0}[data-testid="stSidebar"]>div:first-child{background:transparent}[data-testid="stSidebar"] *{color:#f4f7ff}[data-testid="stSidebar"] .block-container{padding:1rem .75rem 1.25rem}
.sb-brand{display:flex;align-items:center;gap:12px;padding:4px 8px 13px}.sb-logo{width:52px;height:52px;border-radius:16px;display:flex;align-items:center;justify-content:center;background:linear-gradient(145deg,rgba(255,255,255,.23),rgba(255,255,255,.08));border:1px solid rgba(255,255,255,.25);font-size:25px;box-shadow:0 8px 20px rgba(0,0,0,.10)}.sb-name{font-size:18px;font-weight:800;line-height:1.1;color:#fff}.sb-sub{font-size:11px;color:#cbd8fb;margin-top:4px}.sb-profile{display:flex;align-items:center;gap:12px;padding:13px 14px;margin:2px 0 13px;border:1px solid rgba(255,255,255,.20);background:rgba(255,255,255,.10);border-radius:17px}.sb-avatar{width:43px;height:43px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.18);font-size:21px}.sb-profile-name{font-weight:700;font-size:13px;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.sb-role{font-size:11px;color:#cbd8fb;margin-top:3px}.sb-search-label{font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#b8c8ee;margin:8px 4px 5px}[data-testid="stSidebar"] [data-testid="stTextInput"]>div{background:transparent!important;border:0!important;box-shadow:none!important}[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="input"]{background:rgba(255,255,255,.12)!important;border:1px solid rgba(255,255,255,.22)!important;border-radius:13px!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.08)!important;transition:.18s!important}[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="input"]:focus-within{background:rgba(78,129,232,.48)!important;border-color:rgba(143,205,255,.85)!important;box-shadow:0 0 0 3px rgba(91,166,255,.18),0 8px 18px rgba(5,23,75,.16)!important}[data-testid="stSidebar"] [data-testid="stTextInput"] input{background:transparent!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;caret-color:#ffffff!important;border:0!important;outline:none!important;border-radius:13px!important;font-weight:600!important}[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder{color:#d7e4ff!important;opacity:1!important}[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}[data-testid="stSidebar"] [data-testid="stExpander"]>details>div{background:rgba(255,255,255,.035)!important}[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{color:#eef4ff!important}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.09);border-radius:14px;margin:0 0 7px;overflow:hidden}[data-testid="stSidebar"] [data-testid="stExpander"] summary{padding:9px 11px!important;font-weight:800!important}[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{background:rgba(255,255,255,.08)}[data-testid="stSidebar"] .stButton>button{min-height:40px!important;border-radius:11px!important;font-weight:700!important;text-align:left!important;padding:7px 12px!important;margin:2px 0!important;transition:.16s!important}[data-testid="stSidebar"] .stButton>button[kind="secondary"]{background:transparent!important;border:1px solid transparent!important;color:#edf3ff!important}[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover{background:rgba(255,255,255,.11)!important;border-color:rgba(255,255,255,.10)!important;transform:translateX(2px)}[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:linear-gradient(90deg,rgba(104,148,255,.44),rgba(92,128,225,.28))!important;border:1px solid rgba(255,255,255,.18)!important;color:#fff!important;box-shadow:0 7px 18px rgba(5,23,75,.18)!important}[data-testid="stSidebar"] .stButton>button[kind="primary"]:before{content:'●';color:#7ee8ff;margin-right:7px}[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.14)}.sb-location{padding:12px 13px;border-radius:14px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.10)}.sb-location strong{font-size:12px;color:#fff}.sb-location span{display:block;font-size:11px;color:#bfd0f4;margin-top:3px}
/* Login typography */
.login-school-name{font-size:20px;font-weight:800;letter-spacing:.2px;color:#fff}.login-school-sub{font-size:13px;font-weight:600;color:#d9e6ff;margin-top:5px}.login-welcome{position:relative;color:#fff;margin-top:23px!important;font-size:clamp(1.65rem,3vw,2.55rem)!important;line-height:1.12!important;font-weight:800!important;letter-spacing:-.9px!important;max-width:1050px}.login-footer{font-size:12px!important;font-weight:600;color:#7f8ba3!important;letter-spacing:.1px}
/* Sidebar expanders: keep group titles readable in every state */
[data-testid="stSidebar"] [data-testid="stExpander"] details,
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary{background:rgba(255,255,255,.055)!important;color:#fff!important;border-radius:14px!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary{min-height:54px!important;padding:0 14px!important;border:1px solid rgba(255,255,255,.10)!important;font-weight:800!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover{background:rgba(88,139,238,.28)!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] details > summary span{color:#fff!important;-webkit-text-fill-color:#fff!important;font-weight:800!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary{background:linear-gradient(90deg,rgba(80,126,224,.72),rgba(75,116,204,.58))!important;color:#fff!important;border-color:rgba(166,207,255,.35)!important;box-shadow:0 7px 18px rgba(4,22,72,.18)!important}
[data-testid="stSidebar"] [data-testid="stExpander"] details > div{background:rgba(255,255,255,.035)!important;color:#eef4ff!important;border-top:1px solid rgba(255,255,255,.08)!important}
/* Hero */
.hero{position:relative;overflow:hidden;padding:29px 31px;border-radius:24px;margin-bottom:20px;background:linear-gradient(115deg,#18358e 0%,#2d62d0 52%,#42afe8 100%);color:#fff;box-shadow:0 16px 36px rgba(36,81,170,.16);border:1px solid rgba(255,255,255,.22)}.hero:after{content:"";position:absolute;width:300px;height:300px;border-radius:50%;right:-95px;top:-145px;background:rgba(255,255,255,.11)}.hero:before{content:"";position:absolute;width:180px;height:180px;border-radius:50%;left:-80px;bottom:-100px;background:rgba(80,218,211,.12)}.hero h1{position:relative;color:#fff;margin:0;font-size:clamp(1.7rem,3vw,2.35rem);font-weight:800;letter-spacing:-.8px}.hero p{position:relative;color:#eaf2ff;margin:.55rem 0 0;font-size:1rem;line-height:1.7;max-width:950px}.hero-badge{position:relative;display:inline-flex;align-items:center;gap:7px;margin-top:16px;padding:8px 13px;border-radius:999px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.24);color:#fff;font-size:12px;font-weight:700}
/* Cards */
.stat-card,.panel,.category{border:1px solid var(--line);background:#fff;border-radius:20px;box-shadow:var(--shadow);transition:.18s}.stat-card{min-height:130px;padding:20px 21px}.stat-card:hover,.panel:hover,.category:hover{transform:translateY(-2px);box-shadow:var(--shadowHover)}.stat-label{color:var(--muted);font-size:12px;font-weight:700}.stat-value{color:var(--ink);font-size:29px;font-weight:800;margin-top:7px}.stat-note{color:#8b97aa;font-size:11px;margin-top:4px}.section-title{display:flex;align-items:center;gap:8px;font-size:20px;font-weight:800;color:var(--ink);margin:27px 0 13px}.panel{padding:19px 21px}.category{padding:18px}.category .dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px}.dot-low{background:var(--danger)}.dot-mid{background:var(--warning)}.dot-high{background:var(--success)}.small-muted{font-size:12px;color:var(--muted)}.info-strip{padding:13px 16px;border-radius:14px;background:#eef5ff;border:1px solid #d7e4fb;color:#284a8d;font-size:13px;line-height:1.55}.footer-note{font-size:11px;color:#98a2b3;text-align:center;padding:22px 0}
/* Prediction sections */
.fs-head{display:flex;align-items:center;gap:11px;margin-bottom:14px}.fs-icon{width:40px;height:40px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:19px;flex-shrink:0}.fs-title{font-weight:800;font-size:16px;color:var(--ink)}.fs-sub{font-size:12px;color:var(--muted);margin-top:2px}div[class*="st-key-fs_"]{border-radius:20px!important;box-shadow:var(--shadow);margin-bottom:16px;background:#fff!important;border:1px solid var(--line)!important;position:relative;overflow:hidden}div[class*="st-key-fs_"]:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px}div[class*="st-key-fs_akademik"]:before{background:var(--blue700)}div[class*="st-key-fs_lingkungan"]:before{background:var(--cyan)}div[class*="st-key-fs_keluarga"]:before{background:var(--violet)}div[class*="st-key-fs_gaya_hidup"]:before{background:var(--success)}
/* Result */
.result-card{border-radius:24px;padding:28px 30px;margin-top:5px;box-shadow:var(--shadowHover);border:1px solid transparent}.result-high{background:linear-gradient(120deg,#ecfdf5,#effaf8);border-color:#bcebd5}.result-mid{background:linear-gradient(120deg,#fff9e9,#fff5e9);border-color:#f5df9e}.result-low{background:linear-gradient(120deg,#fff1f1,#f9f1ff);border-color:#f3c6c6}.result-eyebrow{font-size:12px;font-weight:700;opacity:.78}.result-title{font-size:30px;font-weight:800;margin:5px 0}.result-meta{display:flex;flex-wrap:wrap;gap:22px;margin-top:13px}.result-meta-item{min-width:140px}.result-meta-label{font-size:11px;font-weight:700;opacity:.72}.result-meta-value{font-size:19px;font-weight:800;margin-top:2px}.result-body{margin-top:17px;padding-top:15px;border-top:1px solid rgba(0,0,0,.08);font-size:13.5px;line-height:1.65}.badge-soft{display:inline-flex;align-items:center;gap:5px;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:700}.badge-low{background:#fef2f2;color:#dc2626}.badge-mid{background:#fffbeb;color:#b77900}.badge-high{background:#ecfdf5;color:#15803d}.activity-item{display:flex;gap:12px;align-items:flex-start;padding:11px 4px;border-bottom:1px solid var(--line)}.activity-item:last-child{border-bottom:0}.activity-icon{width:35px;height:35px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:#f1f5ff;font-size:16px;flex-shrink:0}.activity-title{font-size:13px;font-weight:700;color:var(--ink)}.activity-sub{font-size:11.5px;color:var(--muted);margin-top:2px}
/* Inputs/buttons */
.stButton>button,.stFormSubmitButton>button{border-radius:12px!important;min-height:44px!important;font-weight:700!important;transition:.16s!important}.stButton>button:hover,.stFormSubmitButton>button:hover{box-shadow:var(--shadow)!important;transform:translateY(-1px)}.stFormSubmitButton>button[kind="primary"]{background:linear-gradient(110deg,var(--blue700),var(--violet))!important;color:#fff!important;border:0!important}div[data-baseweb="select"]>div,.stNumberInput input,.stTextInput input,.stTextArea textarea{border-radius:11px!important}[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line)}
@media(max-width:900px){.block-container{padding:.9rem 1rem 2rem}.hero{padding:23px 20px;border-radius:20px}.section-title{font-size:18px}}@media(max-width:650px){.hero h1{font-size:1.5rem}.hero p{font-size:.9rem}.result-card{padding:21px 18px}.result-title{font-size:25px}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE
# ============================================================
# Database mode:
# - Local laptop: SQLite (sistem.db)
# - Online/Streamlit Cloud: PostgreSQL when DATABASE_URL is provided in Secrets
#   This keeps CRUD data persistent even when the Streamlit app restarts.
def get_database_url():
    try:
        if "DATABASE_URL" in st.secrets:
            value = str(st.secrets["DATABASE_URL"]).strip()
            if value:
                return value
    except Exception:
        pass
    return os.environ.get("DATABASE_URL", "").strip()

DATABASE_URL = get_database_url()
USE_POSTGRES = bool(DATABASE_URL)

# Status koneksi ditampilkan kecil agar operator tahu apakah data memakai database online.
if USE_POSTGRES:
    st.markdown('<div style="position:fixed;right:18px;top:12px;z-index:999;font-size:11px;font-weight:700;padding:6px 10px;border-radius:999px;background:#ecfdf5;color:#15803d;border:1px solid #bcebd5">☁️ Database Online Aktif</div>', unsafe_allow_html=True)

class PostgresDB:
    def __init__(self, url):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self._conn = psycopg2.connect(url, sslmode="require")
            self._cursor_factory = RealDictCursor
        except Exception as e:
            raise RuntimeError(
                "Database PostgreSQL belum dapat dihubungkan. Periksa DATABASE_URL pada Streamlit Secrets. "
                f"Detail: {e}"
            ) from e

    def execute(self, sql, params=None):
        sql = sql.replace("?", "%s")
        cur = self._conn.cursor(cursor_factory=self._cursor_factory)
        cur.execute(sql, params or ())
        return cur

    def executescript(self, script):
        # PostgreSQL does not have sqlite3.executescript; execute each statement separately.
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

def conn():
    if USE_POSTGRES:
        return PostgresDB(DATABASE_URL)
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return salt + "$" + digest

def check_password(password, stored):
    try:
        salt, digest = stored.split("$", 1)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
        return secrets.compare_digest(candidate, digest)
    except Exception:
        return False

def init_db():
    c = conn()
    if USE_POSTGRES:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id BIGSERIAL PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, role TEXT, name TEXT
        );
        CREATE TABLE IF NOT EXISTS students(
            id BIGSERIAL PRIMARY KEY, nis TEXT UNIQUE, name TEXT, class_name TEXT, parent_username TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions(
            id BIGSERIAL PRIMARY KEY, username TEXT, nis TEXT, student_name TEXT,
            predicted_class TEXT, probability DOUBLE PRECISION, created_at TEXT, input_json TEXT
        );
        CREATE TABLE IF NOT EXISTS consultations(
            id BIGSERIAL PRIMARY KEY, sender TEXT, receiver TEXT, message TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS development(
            id BIGSERIAL PRIMARY KEY, nis TEXT, student_name TEXT, record_date TEXT,
            category TEXT, note TEXT, created_by TEXT
        );
        """)
    else:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, role TEXT, name TEXT
        );
        CREATE TABLE IF NOT EXISTS students(
            id INTEGER PRIMARY KEY, nis TEXT UNIQUE, name TEXT, class_name TEXT, parent_username TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY, username TEXT, nis TEXT, student_name TEXT,
            predicted_class TEXT, probability REAL, created_at TEXT, input_json TEXT
        );
        CREATE TABLE IF NOT EXISTS consultations(
            id INTEGER PRIMARY KEY, sender TEXT, receiver TEXT, message TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS development(
            id INTEGER PRIMARY KEY, nis TEXT, student_name TEXT, record_date TEXT,
            category TEXT, note TEXT, created_by TEXT
        );
        """)
    for username, password, role, name in DEFAULT_USERS:
        exists = c.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
        if not exists:
            c.execute(
                "INSERT INTO users(username,password_hash,role,name) VALUES(?,?,?,?)",
                (username, hash_password(password), role, name)
            )
    if not c.execute("SELECT 1 FROM students").fetchone():
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for nis, name, class_name, parent_username, predicted_class, probability, source_idx in SEED_STUDENTS:
            c.execute(
                "INSERT INTO students(nis,name,class_name,parent_username,created_at) VALUES(?,?,?,?,?)",
                (nis, name, class_name, parent_username or None, now)
            )
            # Simpan satu riwayat prediksi awal agar 10 data awal langsung terlihat.
            c.execute(
                "INSERT INTO predictions(username,nis,student_name,predicted_class,probability,created_at,input_json) VALUES(?,?,?,?,?,?,?)",
                ("admin", nis, name, predicted_class, probability, now, json.dumps({"seed_source_row": source_idx, "seed_data": True}))
            )
    c.commit(); c.close()

init_db()

# ============================================================
# LOGIN
# ============================================================
def login():
    st.markdown("""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:13px;position:relative">
        <div class="sb-logo">🏫</div>
        <div><div class="login-school-name">SMP GLOBAL PERSADA MANDIRI</div>
        <div class="login-school-sub">Prediksi dan Monitoring Prestasi Akademik Siswa</div></div>
      </div>
      <h1 class="login-welcome">SELAMAT DATANG DI PREDIKSI PRESTASI AKADEMIK SISWA 👋</h1>
      <p>Sistem berbasis web untuk memberikan informasi prediksi prestasi akademik siswa dan mendukung proses monitoring serta tindak lanjut oleh pihak sekolah.</p>
      <div class="hero-badge">🎯 19 variabel prediktor &nbsp;•&nbsp; 📊 3 kategori prestasi &nbsp;•&nbsp; 📍 Bekasi Timur</div>
    </div>
    """, unsafe_allow_html=True)
    left,right=st.columns([1.08,.92],gap="large")
    with left:
        st.markdown('<div class="section-title" style="margin-top:0">🔐 Masuk ke Sistem</div>',unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-bottom:12px">Gunakan akun sesuai peran untuk membuka fitur yang tersedia.</div>',unsafe_allow_html=True)
        with st.form("login_form"):
            username=st.text_input("ID Pengguna",placeholder="Contoh: admin")
            password=st.text_input("Kata Sandi",type="password",placeholder="Masukkan kata sandi")
            submit=st.form_submit_button("Masuk ke Sistem  →",use_container_width=True,type="primary")
        if submit:
            c=conn(); row=c.execute("SELECT * FROM users WHERE username=?",(username.strip(),)).fetchone(); c.close()
            if row and check_password(password,row["password_hash"]):
                st.session_state.user=dict(row); st.session_state.menu="Dashboard"; st.rerun()
            else: st.error("ID pengguna atau kata sandi tidak sesuai.")
        st.markdown('<div class="info-strip" style="margin-top:18px">💡 <b>Tips:</b> gunakan akun pengguna sesuai peran. Setelah masuk, menu <b>Panduan</b> menampilkan daftar akun pengujian.</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title" style="margin-top:0">✨ Fitur Utama</div>',unsafe_allow_html=True)
        features=[("🎯","Prediksi Prestasi","Klasifikasi kategori berdasarkan 19 variabel.","linear-gradient(90deg,#2346a8,#4f83e8)"),("📈","Perkembangan","Riwayat prediksi dan catatan perkembangan siswa.","linear-gradient(90deg,#0ea5a0,#22c7d6)"),("💬","Konsultasi","Media komunikasi antara siswa, orang tua, guru, dan operator.","linear-gradient(90deg,#7255d4,#9a7cf0)"),("📊","Evaluasi Model","Metrik performa dan confusion matrix model.","linear-gradient(90deg,#1498d0,#55b8f3)")]
        for icon,title,desc,grad in features:
            st.markdown(f"""<div class="panel" style="padding:0;overflow:hidden;margin-bottom:12px"><div style="height:7px;background:{grad}"></div><div style="padding:16px 18px;display:flex;gap:12px;align-items:center"><div class="fs-icon" style="background:#eef3ff">{icon}</div><div><b style="color:var(--ink);font-size:14px">{title}</b><div class="small-muted" style="margin-top:3px">{desc}</div></div></div></div>""",unsafe_allow_html=True)
    st.markdown('<div class="footer-note login-footer">Sistem Prediksi Prestasi Akademik Siswa</div>',unsafe_allow_html=True)

if "user" not in st.session_state:
    login()
    st.stop()

u = st.session_state.user
role = u["role"]

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('''
    <div class="sb-brand"><div class="sb-logo">🏫</div><div><div class="sb-name">Prediksi Akademik</div><div class="sb-sub">SMP Global Persada Mandiri</div></div></div>
    ''',unsafe_allow_html=True)
    avatar="👑" if role=="Super Admin" else ("🧑‍🏫" if "Guru" in role else ("🎓" if role=="Siswa" else "👤"))
    st.markdown(f'''<div class="sb-profile"><div class="sb-avatar">{avatar}</div><div><div class="sb-profile-name">{u["name"]}</div><div class="sb-role">{role}</div></div></div>''',unsafe_allow_html=True)
    allowed=ROLES[role]; current=st.session_state.get("menu","Dashboard")
    groups=[("🏠","Utama",["Dashboard"]),("📚","Akademik",["Prediksi Prestasi","Data Siswa"]),("📈","Monitoring",["Perkembangan"]),("💬","Komunikasi",["Konsultasi"]),("📊","Analisis",["Evaluasi Model","Data Penelitian"]),("⚙️","Administrasi",["Daftar Pengguna"]),("❓","Bantuan",["Panduan"])]
    for icon,title,items in groups:
        visible=[x for x in items if x in allowed]
        if not visible: continue
        with st.expander(f"{icon}  {title}",expanded=(current in visible)):
            for item in visible:
                if st.button(item,key=f"nav_{item}",use_container_width=True,type="primary" if item==current else "secondary"):
                    st.session_state.menu=item; st.rerun()
    st.divider()
    st.markdown('<div class="sb-location"><strong>📍 Lokasi pengujian</strong><span>SMP Global Persada Mandiri • Bekasi Timur, Jawa Barat</span></div>',unsafe_allow_html=True)
    if st.button("🚪  Keluar",key="logout_btn",use_container_width=True,type="secondary"):
        st.session_state.pop("user",None); st.rerun()

# ============================================================
# MENU AKTIF
# ============================================================
# Sidebar menyimpan halaman aktif di session_state. Variabel lokal
# harus diambil kembali sebelum routing halaman dijalankan.
menu = st.session_state.get("menu", "Dashboard")

# ============================================================
# HELPERS
# ============================================================
def hero(title, subtitle, badge=None):
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ""
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{badge_html}</div>', unsafe_allow_html=True)

def metric_card(label, value, note=""):
    return f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>'

def nice_columns(df):
    rename = {
        "created_at":"Tanggal","nis":"NIS","student_name":"Nama Siswa",
        "predicted_class":"Hasil Prediksi","probability":"Probabilitas",
        "name":"Nama Siswa","class_name":"Kelas","parent_username":"ID Orang Tua/Wali",
        "record_date":"Tanggal","category":"Kategori","note":"Catatan","created_by":"Dicatat Oleh",
        "sender":"Pengirim","receiver":"Penerima","message":"Pesan",
        "username":"ID Pengguna","role":"Peran",
    }
    return df.rename(columns=rename)

def target_badge(pred):
    if pred == "Rendah":
        return "🔴", "result-low"
    if pred == "Sedang":
        return "🟡", "result-mid"
    return "🟢", "result-high"

def form_section_head(icon, title, sub, accent_bg):
    st.markdown(
        f'<div class="fs-head"><div class="fs-icon" style="background:{accent_bg}">{icon}</div>'
        f'<div><div class="fs-title">{title}</div><div class="fs-sub">{sub}</div></div></div>',
        unsafe_allow_html=True
    )

def prediction_form():
    v = {}
    # 1) DATA AKADEMIK — Hours_Studied, Attendance, Previous_Scores, Tutoring_Sessions
    with st.container(border=True, key="fs_akademik"):
        form_section_head("📚","Data Akademik","Riwayat belajar dan pencapaian akademik siswa","#EEF3FF")
        c1,c2 = st.columns(2)
        with c1:
            v["Hours_Studied"] = st.number_input("Jam belajar / minggu", 0.0, 100.0, float(meta["numeric_stats"]["Hours_Studied"]["median"]), 1.0)
            v["Previous_Scores"] = st.number_input("Nilai sebelumnya", 0.0, 100.0, float(meta["numeric_stats"]["Previous_Scores"]["median"]), 1.0)
        with c2:
            v["Attendance"] = st.number_input("Kehadiran (%)", 0.0, 100.0, float(meta["numeric_stats"]["Attendance"]["median"]), 1.0)
            v["Tutoring_Sessions"] = st.number_input("Sesi bimbingan", 0.0, 20.0, float(meta["numeric_stats"]["Tutoring_Sessions"]["median"]), 1.0)

    # 2) LINGKUNGAN BELAJAR — Access_to_Resources, Internet_Access, Teacher_Quality, School_Type, Distance_from_Home
    with st.container(border=True, key="fs_lingkungan"):
        form_section_head("🏫","Lingkungan Belajar","Fasilitas, akses, dan kondisi sekolah/tempat belajar","#EAF8FE")
        c1,c2,c3 = st.columns(3)
        with c1:
            v["Access_to_Resources"] = st.selectbox("Akses ke Sumber Daya", meta["options"]["Access_to_Resources"])
            v["Internet_Access"] = st.selectbox("Akses Internet", meta["options"]["Internet_Access"])
        with c2:
            v["Teacher_Quality"] = st.selectbox("Kualitas Guru", meta["options"]["Teacher_Quality"])
            v["School_Type"] = st.selectbox("Jenis Sekolah", meta["options"]["School_Type"])
        with c3:
            v["Distance_from_Home"] = st.selectbox("Jarak dari Rumah", meta["options"]["Distance_from_Home"])

    # 3) DUKUNGAN KELUARGA & SOSIAL — Parental_Involvement, Family_Income, Parental_Education_Level, Peer_Influence
    with st.container(border=True, key="fs_keluarga"):
        form_section_head("👨‍👩‍👧","Dukungan Keluarga & Sosial","Keterlibatan keluarga dan pengaruh lingkungan sosial","#F3EEFF")
        c1,c2 = st.columns(2)
        with c1:
            v["Parental_Involvement"] = st.selectbox("Keterlibatan Orang Tua", meta["options"]["Parental_Involvement"])
            v["Family_Income"] = st.selectbox("Pendapatan Keluarga", meta["options"]["Family_Income"])
        with c2:
            v["Parental_Education_Level"] = st.selectbox("Pendidikan Orang Tua", meta["options"]["Parental_Education_Level"])
            v["Peer_Influence"] = st.selectbox("Pengaruh Teman Sebaya", meta["options"]["Peer_Influence"])

    # 4) GAYA HIDUP & KARAKTERISTIK — Extracurricular_Activities, Sleep_Hours, Motivation_Level, Physical_Activity, Learning_Disabilities, Gender
    with st.container(border=True, key="fs_gaya_hidup"):
        form_section_head("🌙","Gaya Hidup & Karakteristik","Kebiasaan sehari-hari dan karakteristik pribadi siswa","#EAFBF8")
        c1,c2,c3 = st.columns(3)
        with c1:
            v["Sleep_Hours"] = st.number_input("Jam tidur", 0.0, 24.0, float(meta["numeric_stats"]["Sleep_Hours"]["median"]), 1.0)
            v["Physical_Activity"] = st.number_input("Aktivitas fisik / minggu", 0.0, 20.0, float(meta["numeric_stats"]["Physical_Activity"]["median"]), 1.0)
        with c2:
            v["Extracurricular_Activities"] = st.selectbox("Kegiatan Ekstrakurikuler", meta["options"]["Extracurricular_Activities"])
            v["Motivation_Level"] = st.selectbox("Tingkat Motivasi", meta["options"]["Motivation_Level"])
        with c3:
            v["Learning_Disabilities"] = st.selectbox("Kesulitan Belajar", meta["options"]["Learning_Disabilities"])
            v["Gender"] = st.selectbox("Jenis Kelamin", meta["options"]["Gender"])
    return v

# ============================================================
# DASHBOARD
# ============================================================
if menu == "Dashboard":
    if role == "Orang Tua/Wali":
        dashboard_title = "Selamat Datang, Orang Tua/Wali Murid 👋"
    elif role == "Siswa":
        dashboard_title = "Hello, Selamat Datang Siswa SMP Global Persada Mandiri 👋"
    else:
        dashboard_title = f"Selamat Datang, {u['name']} 👋"

    hero(
        dashboard_title,
        f"Dashboard {role} untuk memantau informasi dan prediksi prestasi akademik.",
        f"📌 {len(meta['predictors'])} variabel prediktor • Gaussian Naive Bayes • data uji 20%"
    )
    m = meta["metrics"]
    cols = st.columns(4)
    cards = [
        ("Data Model", f"{meta['dataset_rows']:,}".replace(",","."), "baris dataset sekunder"),
        ("Prediktor", str(len(meta["predictors"])), "variabel input model"),
        ("Accuracy", f"{m['accuracy']:.2%}", "hasil pada data uji"),
        ("F1-Score", f"{m['f1_score']:.2%}", "weighted"),
    ]
    for col,(a,b,c) in zip(cols,cards):
        with col: st.markdown(metric_card(a,b,c), unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Ringkasan Kategori Prediksi</div>', unsafe_allow_html=True)
    dist = meta["class_distribution"]
    total = sum(dist.values())
    c1,c2,c3 = st.columns(3)
    cats = [
        ("Rendah","≤66",dist["Rendah"],"dot-low"),
        ("Sedang","67–69",dist["Sedang"],"dot-mid"),
        ("Tinggi","≥70",dist["Tinggi"],"dot-high")
    ]
    for col,(cat,rng,num,dot) in zip([c1,c2,c3],cats):
        with col:
            pct = num/total if total else 0
            st.markdown(
                f'<div class="category"><span class="dot {dot}"></span><b>{cat}</b>'
                f'<div style="font-size:27px;font-weight:850;margin-top:8px">{num:,}</div>'
                f'<div class="small-muted">{pct:.2%} dari dataset • rentang kategori {rng}</div></div>'.replace(",","."),
                unsafe_allow_html=True
            )
    st.markdown('<div class="info-strip" style="margin-top:14px">ℹ️ Kategori prediksi ditentukan berdasarkan <i>Exam_Score</i> pada dataset model dan bukan merupakan standar penilaian resmi sekolah.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 Distribusi Data</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame({"Jumlah": [dist["Rendah"],dist["Sedang"],dist["Tinggi"]]}, index=["Rendah","Sedang","Tinggi"])
    st.bar_chart(chart_df, height=300)
    st.caption("Gunakan grafik interaktif untuk melihat distribusi data pada setiap kategori prediksi.")

    st.markdown('<div class="section-title">🧭 Informasi Sistem</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><b>Metode</b><br><span class="small-muted">CRISP-DM • Gaussian Naive Bayes</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><b>Pembagian Data</b><br><span class="small-muted">80% data latih • 20% data uji • random_state 42</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><b>Lokasi Pengujian</b><br><span class="small-muted">SMP Global Persada Mandiri • Bekasi</span></div>', unsafe_allow_html=True)

    # --- Ringkasan Prediksi & Aktivitas Terbaru ---
    # Scoping mengikuti prinsip least privilege yang sama dengan halaman Perkembangan:
    # Siswa hanya melihat datanya sendiri, Orang Tua/Wali hanya siswa yang terhubung,
    # role lain tidak dibatasi.
    dc = conn()
    if role == "Siswa":
        dash_scope_nis = [u["username"]]
    elif role == "Orang Tua/Wali":
        _linked = dc.execute("SELECT nis FROM students WHERE parent_username=?", (u["username"],)).fetchall()
        dash_scope_nis = [r["nis"] for r in _linked]
    else:
        dash_scope_nis = None

    if dash_scope_nis is not None:
        if dash_scope_nis:
            ph = ",".join("?"*len(dash_scope_nis))
            recent_pred = dc.execute(f"SELECT student_name,nis,predicted_class,probability,created_at FROM predictions WHERE nis IN ({ph}) ORDER BY id DESC LIMIT 5", dash_scope_nis).fetchall()
            recent_dev = dc.execute(f"SELECT student_name,category,record_date AS created_at FROM development WHERE nis IN ({ph}) ORDER BY id DESC LIMIT 5", dash_scope_nis).fetchall()
        else:
            recent_pred, recent_dev = [], []
    else:
        recent_pred = dc.execute("SELECT student_name,nis,predicted_class,probability,created_at FROM predictions ORDER BY id DESC LIMIT 5").fetchall()
        recent_dev = dc.execute("SELECT student_name,category,record_date AS created_at FROM development ORDER BY id DESC LIMIT 5").fetchall()
    recent_msg = dc.execute("SELECT sender,receiver,message,created_at FROM consultations WHERE sender=? OR receiver=? ORDER BY id DESC LIMIT 5", (u["username"], u["username"])).fetchall()
    dc.close()

    badge_cls = {"Rendah":"badge-low","Sedang":"badge-mid","Tinggi":"badge-high"}
    left_col, right_col = st.columns([1.1, 1], gap="large")
    with left_col:
        st.markdown('<div class="section-title">🕘 Ringkasan Prediksi Terbaru</div>', unsafe_allow_html=True)
        pred_search = st.text_input(
            "Cari prediksi siswa",
            placeholder="🔎 Ketik nama siswa atau NIS...",
            key="dashboard_pred_search"
        )
        pred_q = pred_search.strip().lower()
        filtered_pred = [r for r in recent_pred if not pred_q or pred_q in str(r["student_name"]).lower() or pred_q in str(r["nis"]).lower() or pred_q in str(r["predicted_class"]).lower()]
        if filtered_pred:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            for r in filtered_pred:
                bcls = badge_cls.get(r["predicted_class"], "badge-mid")
                st.markdown(
                    f'<div class="activity-item"><div class="activity-icon">🎯</div>'
                    f'<div style="flex:1"><div class="activity-title">{r["student_name"]} <span class="small-muted">({r["nis"]})</span></div>'
                    f'<div class="activity-sub">{r["created_at"]}</div></div>'
                    f'<span class="badge-soft {bcls}">{r["predicted_class"]} • {float(r["probability"]):.0%}</span></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        elif recent_pred:
            st.info("Tidak ada prediksi yang cocok dengan pencarian.")
        else:
            st.markdown('<div class="info-strip">Belum ada riwayat prediksi untuk ditampilkan.</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-title">📋 Aktivitas Terbaru</div>', unsafe_allow_html=True)
        activity_search = st.text_input(
            "Cari aktivitas",
            placeholder="🔎 Cari nama siswa, aktivitas, kategori, atau pesan...",
            key="dashboard_activity_search"
        )
        feed = []
        for r in recent_pred:
            feed.append((r["created_at"], "🎯", f"Prediksi untuk {r['student_name']}", f"Hasil: {r['predicted_class']}"))
        for r in recent_dev:
            feed.append((r["created_at"], "📈", f"Catatan perkembangan — {r['student_name']}", f"Kategori: {r['category']}"))
        for r in recent_msg:
            arah = "Terkirim" if r["sender"] == u["username"] else "Diterima"
            feed.append((r["created_at"], "💬", f"Pesan konsultasi ({arah})", r["message"][:60] + ("…" if len(r["message"])>60 else "")))
        feed.sort(key=lambda x: x[0], reverse=True)
        activity_q = activity_search.strip().lower()
        filtered_feed = [x for x in feed if not activity_q or any(activity_q in str(v).lower() for v in x)]
        filtered_feed = filtered_feed[:5]
        if filtered_feed:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            for ts, icon, title, sub in filtered_feed:
                st.markdown(
                    f'<div class="activity-item"><div class="activity-icon">{icon}</div>'
                    f'<div><div class="activity-title">{title}</div>'
                    f'<div class="activity-sub">{sub} • {ts}</div></div></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        elif feed:
            st.info("Tidak ada aktivitas yang cocok dengan pencarian.")
        else:
            st.markdown('<div class="info-strip">Belum ada aktivitas untuk ditampilkan.</div>', unsafe_allow_html=True)

# ============================================================
# PREDIKSI
# ============================================================
elif menu == "Prediksi Prestasi":
    hero("🔎 Prediksi Prestasi Siswa", f"Masukkan {len(meta['predictors'])} variabel prediktor untuk memperoleh kategori hasil prediksi dan probabilitas model.", "🎯 Hasil: Rendah • Sedang • Tinggi")
    c = conn()
    # Least privilege: role Siswa hanya boleh memprediksi untuk dirinya sendiri
    # (akun Siswa dibuat dengan username == NIS, lihat data pengguna awal pada proses inisialisasi).
    # Role lain (Operator/Guru/Wali Kelas/Super Admin) tetap melihat seluruh daftar siswa
    # sesuai kewenangannya masing-masing (tidak diubah).
    if role == "Siswa":
        students = c.execute("SELECT nis,name FROM students WHERE nis=?", (u["username"],)).fetchall()
    else:
        students = c.execute("SELECT nis,name FROM students ORDER BY name").fetchall()
    c.close()
    if not students:
        if role == "Siswa":
            st.warning("Data siswa untuk akun ini belum terdaftar. Hubungi operator/guru untuk menambahkan data siswa Anda.")
        else:
            st.warning("Belum ada data siswa. Tambahkan siswa terlebih dahulu pada menu Data Siswa.")
        st.stop()
    if role == "Siswa":
        # Tidak menampilkan dropdown pilihan siswa lain — otomatis terkunci ke akun sendiri
        r0 = students[0]
        nis, sname = r0["nis"], r0["name"]
        st.markdown(f'<div class="info-strip">👤 Prediksi akan dijalankan untuk akun Anda: <b>{sname} ({nis})</b></div>', unsafe_allow_html=True)
    else:
        opts = {f"{r['nis']} — {r['name']}": r["nis"] for r in students}
        selected = st.selectbox("👨‍🎓 Pilih siswa", list(opts.keys()), key="predict_student_select")
        nis = opts[selected]
        sname = selected.split(" — ",1)[1]

    with st.form("prediction_form"):
        values = prediction_form()
        run = st.form_submit_button("🚀 Jalankan Prediksi", use_container_width=True)

    if run:
        X = pre.transform(pd.DataFrame([values]))
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        class_order = list(model.classes_)
        top = float(probs[class_order.index(pred)])
        icon, cls = target_badge(pred)

        ranges = {"Rendah":"≤66", "Sedang":"67–69", "Tinggi":"≥70"}
        recommendations = {
            "Rendah": "Perlu penguatan konsistensi belajar, kehadiran, dan pendampingan.",
            "Sedang": "Pertahankan kebiasaan belajar dan tingkatkan faktor pendukung secara bertahap.",
            "Tinggi": "Pertahankan pola belajar dan faktor pendukung yang sudah baik."
        }
        st.markdown(
            f'<div class="result-card {cls}">'
            f'<div class="result-eyebrow">🎓 Hasil Prediksi — {sname} ({nis})</div>'
            f'<div class="result-title">{icon} {pred.upper()}</div>'
            f'<div class="result-meta">'
            f'<div class="result-meta-item"><div class="result-meta-label">RENTANG KATEGORI</div><div class="result-meta-value">{ranges[pred]}</div></div>'
            f'<div class="result-meta-item"><div class="result-meta-label">PROBABILITAS MODEL</div><div class="result-meta-value">{top:.2%}</div></div>'
            f'<div class="result-meta-item"><div class="result-meta-label">MODEL</div><div class="result-meta-value" style="font-size:15px">Gaussian Naive Bayes</div></div>'
            f'</div>'
            f'<div class="result-body">'
            f'<b>Interpretasi:</b> berdasarkan karakteristik data yang dimasukkan, model mengklasifikasikan siswa ke dalam kategori prestasi <b>{pred}</b> ({ranges[pred]}). Hasil merupakan prediksi kategori, bukan nilai ujian yang exact.<br><br>'
            f'<b>Rekomendasi umum:</b> {recommendations[pred]}'
            f'</div></div>',
            unsafe_allow_html=True
        )

        result_df = pd.DataFrame({
            "Kategori": class_order,
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)
        result_df["Probabilitas"] = result_df["Probabilitas"].map(lambda x: f"{x:.2%}")
        st.markdown('<div class="section-title">📌 Probabilitas Setiap Kategori</div>', unsafe_allow_html=True)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        c = conn()
        c.execute(
            "INSERT INTO predictions(username,nis,student_name,predicted_class,probability,created_at,input_json) VALUES(?,?,?,?,?,?,?)",
            (u["username"],nis,sname,pred,top,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),json.dumps(values))
        )
        c.commit(); c.close()
        st.success("Hasil prediksi berhasil disimpan ke riwayat.")

# ============================================================
# DATA SISWA
# ============================================================
elif menu == "Data Siswa":
    hero("👨‍🎓 Data Siswa", "Kelola data operasional siswa. Setiap penambahan, perubahan, dan penghapusan langsung disimpan ke database.", "💾 Perubahan tersimpan otomatis saat tombol disimpan")

    c = conn()
    rows = c.execute("""
        SELECT s.id, s.nis, s.name, s.class_name, COALESCE(s.parent_username, '-') AS parent_username,
               COALESCE((SELECT p.predicted_class FROM predictions p WHERE p.nis=s.nis ORDER BY p.id DESC LIMIT 1), 'Belum Diprediksi') AS predicted_class,
               COALESCE((SELECT p.probability FROM predictions p WHERE p.nis=s.nis ORDER BY p.id DESC LIMIT 1), 0) AS probability,
               s.created_at
        FROM students s ORDER BY s.name
    """).fetchall()
    c.close()

    # Ringkasan jumlah data berdasarkan prediksi terakhir.
    counts = {"Rendah": 0, "Sedang": 0, "Tinggi": 0}
    for r in rows:
        if r["predicted_class"] in counts:
            counts[r["predicted_class"]] += 1
    a,b,c1,d = st.columns(4)
    with a: st.metric("Total Siswa", len(rows))
    with b: st.metric("🔴 Rendah", counts["Rendah"])
    with c1: st.metric("🟡 Sedang", counts["Sedang"])
    with d: st.metric("🟢 Tinggi", counts["Tinggi"])

    display_rows = []
    for r in rows:
        display_rows.append({
            "NIS": r["nis"], "Nama Siswa": r["name"], "Kelas": r["class_name"],
            "ID Orang Tua/Wali": r["parent_username"], "Prediksi Terakhir": r["predicted_class"],
            "Probabilitas": f'{float(r["probability"]):.2%}' if r["predicted_class"] != "Belum Diprediksi" else "-",
            "Tanggal Ditambahkan": pd.to_datetime(r["created_at"], errors="coerce").strftime("%d-%m-%Y %H:%M") if pd.notna(pd.to_datetime(r["created_at"], errors="coerce")) else r["created_at"]
        })
    df_students = pd.DataFrame(display_rows)
    st.dataframe(df_students, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # Tambah siswa
    # --------------------------------------------------------
    st.markdown('<div class="section-title">➕ Tambah Data Siswa</div>', unsafe_allow_html=True)
    with st.form("add_student"):
        a,b,c1 = st.columns(3)
        with a: nis_new = st.text_input("NIS", placeholder="Contoh: NIS011")
        with b: name_new = st.text_input("Nama Siswa", placeholder="Nama lengkap")
        with c1: class_new = st.text_input("Kelas", placeholder="Contoh: VIII A")
        parent_new = st.text_input("ID Orang Tua/Wali (opsional)", placeholder="Contoh: ORTU001")
        add = st.form_submit_button("💾 Simpan Data Siswa", use_container_width=True)
    if add:
        if not all([nis_new.strip(), name_new.strip(), class_new.strip()]):
            st.error("NIS, nama siswa, dan kelas wajib diisi.")
        else:
            try:
                c = conn()
                c.execute(
                    "INSERT INTO students(nis,name,class_name,parent_username,created_at) VALUES(?,?,?,?,?)",
                    (nis_new.strip(), name_new.strip(), class_new.strip(), parent_new.strip() or None, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                c.commit(); c.close()
                st.success(f"Data {name_new.strip()} berhasil disimpan. Data akan tetap tersimpan meskipun aplikasi ditutup.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("NIS sudah terdaftar. Gunakan NIS yang berbeda.")

    # --------------------------------------------------------
    # Edit siswa
    # --------------------------------------------------------
    st.markdown('<div class="section-title">✏️ Edit Data Siswa</div>', unsafe_allow_html=True)
    if rows:
        edit_opts = {f'{r["nis"]} — {r["name"]}': r["id"] for r in rows}
        selected_edit = st.selectbox("Pilih siswa yang ingin diedit", list(edit_opts.keys()), key="edit_student_select")
        edit_id = edit_opts[selected_edit]
        current = next(r for r in rows if r["id"] == edit_id)
        with st.form("edit_student_form"):
            a,b,c1 = st.columns(3)
            with a: nis_edit = st.text_input("NIS", value=current["nis"])
            with b: name_edit = st.text_input("Nama Siswa", value=current["name"])
            with c1: class_edit = st.text_input("Kelas", value=current["class_name"])
            parent_edit = st.text_input("ID Orang Tua/Wali (opsional)", value=current["parent_username"] if current["parent_username"] != "-" else "")
            save_edit = st.form_submit_button("💾 Simpan Perubahan", use_container_width=True)
        if save_edit:
            if not all([nis_edit.strip(), name_edit.strip(), class_edit.strip()]):
                st.error("NIS, nama siswa, dan kelas wajib diisi.")
            else:
                try:
                    c = conn()
                    c.execute(
                        "UPDATE students SET nis=?, name=?, class_name=?, parent_username=? WHERE id=?",
                        (nis_edit.strip(), name_edit.strip(), class_edit.strip(), parent_edit.strip() or None, edit_id)
                    )
                    # Sinkronkan nama/NIS pada riwayat prediksi dan perkembangan agar tidak menyisakan data lama.
                    c.execute("UPDATE predictions SET nis=?, student_name=? WHERE nis=?", (nis_edit.strip(), name_edit.strip(), current["nis"]))
                    c.execute("UPDATE development SET nis=?, student_name=? WHERE nis=?", (nis_edit.strip(), name_edit.strip(), current["nis"]))
                    c.commit(); c.close()
                    st.success("Perubahan data berhasil disimpan otomatis ke database.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("NIS baru sudah digunakan oleh siswa lain.")
    else:
        st.info("Belum ada data siswa untuk diedit.")

    # --------------------------------------------------------
    # Hapus siswa
    # --------------------------------------------------------
    st.markdown('<div class="section-title">🗑️ Hapus Data Siswa</div>', unsafe_allow_html=True)
    if rows:
        delete_opts = {f'{r["nis"]} — {r["name"]}': r["id"] for r in rows}
        selected_delete = st.selectbox("Pilih siswa yang ingin dihapus", list(delete_opts.keys()), key="delete_student_select")
        delete_id = delete_opts[selected_delete]
        delete_row = next(r for r in rows if r["id"] == delete_id)
        confirm_delete = st.checkbox(f"Saya yakin ingin menghapus data {delete_row['name']} ({delete_row['nis']}) beserta riwayat prediksinya.", key="confirm_delete_student")
        if st.button("🗑️ Hapus Data Siswa", type="secondary", use_container_width=True, disabled=not confirm_delete):
            c = conn()
            c.execute("DELETE FROM predictions WHERE nis=?", (delete_row["nis"],))
            c.execute("DELETE FROM development WHERE nis=?", (delete_row["nis"],))
            c.execute("DELETE FROM students WHERE id=?", (delete_id,))
            c.commit(); c.close()
            st.success(f"Data {delete_row['name']} berhasil dihapus dan perubahan sudah tersimpan.")
            st.rerun()
    else:
        st.info("Belum ada data siswa untuk dihapus.")

    st.markdown('<div class="info-strip" style="margin-top:18px">💡 <b>Catatan penyimpanan:</b> pada penggunaan lokal di laptop, database <b>sistem.db</b> tersimpan di folder aplikasi sehingga data tidak hilang ketika browser ditutup atau aplikasi dijalankan kembali. Untuk versi online/cloud, database perlu dipindahkan ke database cloud persisten agar data tetap aman setelah aplikasi restart.</div>', unsafe_allow_html=True)

# ============================================================
# PERKEMBANGAN
# ============================================================
elif menu == "Perkembangan":
    hero("📈 Perkembangan Akademik", "Riwayat prediksi dan catatan perkembangan yang tersimpan pada sistem.", "📝 Catatan operasional • bukan data longitudinal dari dataset Kaggle")
    c = conn()
    # Least privilege — filter dilakukan di level query (SQL), bukan disembunyikan di UI:
    # - Siswa: hanya baris dengan nis miliknya sendiri (username == nis)
    # - Orang Tua/Wali: hanya baris untuk NIS yang terhubung ke akunnya (students.parent_username)
    # - Guru/Wali Kelas, Operator, Super Admin: tidak dibatasi (sesuai kewenangan saat ini,
    #   karena skema data belum memiliki pemetaan kelas/guru per siswa)
    if role == "Siswa":
        scope_nis = [u["username"]]
    elif role == "Orang Tua/Wali":
        linked = c.execute("SELECT nis FROM students WHERE parent_username=?", (u["username"],)).fetchall()
        scope_nis = [row["nis"] for row in linked]
    else:
        scope_nis = None  # tanpa batasan

    if scope_nis is not None:
        if scope_nis:
            placeholders = ",".join("?" * len(scope_nis))
            pr = c.execute(
                f"SELECT created_at,nis,student_name,predicted_class,probability FROM predictions WHERE nis IN ({placeholders}) ORDER BY id DESC",
                scope_nis
            ).fetchall()
            dv = c.execute(
                f"SELECT record_date,nis,student_name,category,note,created_by FROM development WHERE nis IN ({placeholders}) ORDER BY id DESC",
                scope_nis
            ).fetchall()
        else:
            pr, dv = [], []
    else:
        pr = c.execute("SELECT created_at,nis,student_name,predicted_class,probability FROM predictions ORDER BY id DESC").fetchall()
        dv = c.execute("SELECT record_date,nis,student_name,category,note,created_by FROM development ORDER BY id DESC").fetchall()
    c.close()

    if role == "Orang Tua/Wali" and not scope_nis:
        st.info("Belum ada data siswa yang terhubung dengan akun Orang Tua/Wali ini.")

    st.markdown('<div class="section-title">🕘 Riwayat Prediksi</div>', unsafe_allow_html=True)
    if pr:
        pdf = nice_columns(pd.DataFrame([dict(x) for x in pr]))
        pdf["Tanggal"] = pd.to_datetime(pdf["Tanggal"]).dt.strftime("%d-%m-%Y %H:%M")
        pdf["Probabilitas"] = pdf["Probabilitas"].map(lambda x: f"{float(x):.2%}")
        st.dataframe(pdf, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada riwayat prediksi.")

    if role in ["Super Admin","Operator","Guru/Wali Kelas"]:
        st.markdown('<div class="section-title">📝 Tambah Catatan Perkembangan</div>', unsafe_allow_html=True)
        with st.form("development_form"):
            a,b,c = st.columns(3)
            with a: rec_date = st.date_input("Tanggal", value=datetime.now().date())
            with b: rec_nis = st.text_input("NIS", value="NIS001")
            with c: rec_name = st.text_input("Nama Siswa", value="Siswa Baru")
            category = st.selectbox("Kategori", ["Rendah","Sedang","Tinggi"])
            note = st.text_area("Catatan", placeholder="Tuliskan catatan perkembangan secara singkat...")
            save = st.form_submit_button("Simpan Catatan")
        if save:
            if not rec_nis.strip() or not rec_name.strip():
                st.error("NIS dan nama siswa wajib diisi.")
            else:
                c = conn()
                c.execute("INSERT INTO development(nis,student_name,record_date,category,note,created_by) VALUES(?,?,?,?,?,?)",
                          (rec_nis.strip(),rec_name.strip(),str(rec_date),category,note.strip(),u["username"]))
                c.commit(); c.close()
                st.success("Catatan perkembangan berhasil disimpan.")
                st.rerun()

    st.markdown('<div class="section-title">📋 Catatan Perkembangan</div>', unsafe_allow_html=True)
    if dv:
        ddf = nice_columns(pd.DataFrame([dict(x) for x in dv]))
        ddf["Tanggal"] = pd.to_datetime(ddf["Tanggal"]).dt.strftime("%d-%m-%Y")
        st.dataframe(ddf, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada catatan.")

# ============================================================
# KONSULTASI
# ============================================================
elif menu == "Konsultasi":
    hero("💬 Konsultasi", "Media komunikasi pendukung antara siswa/orang tua dengan guru atau operator.", "🤝 Fitur pendukung sistem")
    c = conn()
    recipients = c.execute("SELECT username,name,role FROM users WHERE role IN ('Guru/Wali Kelas','Operator','Kepala Sekolah') ORDER BY role,name").fetchall()
    msgs = c.execute("SELECT sender,receiver,message,created_at FROM consultations WHERE sender=? OR receiver=? ORDER BY id DESC",(u["username"],u["username"])).fetchall()
    c.close()

    if role in ["Siswa","Orang Tua/Wali"]:
        choices = {f"{x['name']} — {x['role']}":x["username"] for x in recipients}
        with st.form("consult_form"):
            to_label = st.selectbox("Kirim kepada", list(choices.keys()))
            message = st.text_area("Pesan", placeholder="Tuliskan pertanyaan atau informasi...")
            send = st.form_submit_button("📨 Kirim Pesan", use_container_width=True)
        if send:
            if not message.strip():
                st.error("Pesan tidak boleh kosong.")
            else:
                c = conn()
                c.execute("INSERT INTO consultations(sender,receiver,message,created_at) VALUES(?,?,?,?)",
                          (u["username"],choices[to_label],message.strip(),datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                c.commit(); c.close()
                st.success("Pesan berhasil dikirim.")
                st.rerun()
    else:
        st.info("Gunakan tabel di bawah untuk melihat percakapan yang terkait dengan akun ini.")

    if msgs:
        mdf = nice_columns(pd.DataFrame([dict(x) for x in msgs]))
        mdf["Tanggal"] = pd.to_datetime(mdf["Tanggal"]).dt.strftime("%d-%m-%Y %H:%M")
        st.dataframe(mdf, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada pesan.")

# ============================================================
# EVALUASI MODEL
# ============================================================
elif menu == "Evaluasi Model":
    hero("📊 Evaluasi Model", "Hasil pengujian Gaussian Naive Bayes menggunakan data uji 20% dengan pemisahan berstrata.", "🧪 Held-out test set • random_state 42")
    m = meta["metrics"]
    cols = st.columns(4)
    cards = [
        ("Accuracy",f"{m['accuracy']:.2%}","ketepatan klasifikasi"),
        ("Precision",f"{m['precision']:.2%}","weighted"),
        ("Recall",f"{m['recall']:.2%}","weighted"),
        ("F1-Score",f"{m['f1_score']:.2%}","weighted"),
    ]
    for col,(a,b,c) in zip(cols,cards):
        with col: st.markdown(metric_card(a,b,c),unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔢 Matriks Konfusi</div>', unsafe_allow_html=True)
    cm = pd.DataFrame(
        meta["confusion_matrix"],
        index=["Aktual Rendah","Aktual Sedang","Aktual Tinggi"],
        columns=["Prediksi Rendah","Prediksi Sedang","Prediksi Tinggi"]
    )
    st.dataframe(cm, use_container_width=True)
    st.markdown(
        f'<div class="info-strip">⚙️ <b>var_smoothing:</b> {meta["tuning"]["var_smoothing"]} &nbsp; • &nbsp; '
        f'<b>Data latih:</b> {meta["train_rows"]:,} &nbsp; • &nbsp; <b>Data uji:</b> {meta["test_rows"]:,}</div>'.replace(",","."),
        unsafe_allow_html=True
    )

# ============================================================
# DATA PENELITIAN
# ============================================================
elif menu == "Data Penelitian":
    hero("📚 Data Penelitian", "Informasi dataset, variabel prediktor, target klasifikasi, dan praproses data.", "🗃️ Dataset sekunder • Student Performance Factors")
    st.markdown(
        '<div class="info-strip">📌 Dataset penelitian merupakan data sekunder <b>Student Performance Factors</b> yang diperoleh dari <b>Kaggle</b> dan <b>BUKAN</b> merupakan data siswa SMP Global Persada Mandiri.<br>'
        'Dataset digunakan untuk membangun dan menguji model prediksi, sedangkan SMP Global Persada Mandiri merupakan lokasi pelaksanaan penelitian dan uji coba sistem.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">📌 Variabel Prediktor</div>', unsafe_allow_html=True)
    rows = []
    for i,col in enumerate(meta["predictors"],1):
        rows.append([i,col,"Numerik" if col in meta["numeric_features"] else "Kategorikal"])
    var_df = pd.DataFrame(rows,columns=["No","Variabel","Jenis"])
    st.dataframe(var_df,use_container_width=True,hide_index=True)

    st.markdown('<div class="section-title">🧹 Praproses Data</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><b>Nilai yang hilang</b><br><span class="small-muted">Teacher_Quality → Medium<br>Parental_Education_Level → High School<br>Distance_from_Home → Near</span></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><b>Pengkodean</b><br><span class="small-muted">Variabel kategorikal → one-hot encoding<br>Variabel numerik → tetap numerik<br>Feature selection → tidak digunakan</span></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Aturan Target</div>', unsafe_allow_html=True)
    target_df = pd.DataFrame({
        "Kategori":["Rendah","Sedang","Tinggi"],
        "Rentang":["≤66","67–69","≥70"],
        "Jumlah":[meta["class_distribution"]["Rendah"],meta["class_distribution"]["Sedang"],meta["class_distribution"]["Tinggi"]]
    })
    st.dataframe(target_df,use_container_width=True,hide_index=True)
    st.markdown('<div class="info-strip">ℹ️ Rentang tersebut adalah aturan penelitian berdasarkan <i>Exam_Score</i> pada dataset sekunder, bukan standar penilaian resmi sekolah.</div>',unsafe_allow_html=True)

# ============================================================
# MANAJEMEN PENGGUNA
# ============================================================
elif menu == "Daftar Pengguna":
    hero("👥 Daftar Pengguna", "Daftar akun dan pembagian hak akses sistem (tampilan saja, tanpa fitur tambah/ubah/hapus akun).", "🔐 Password tersimpan dalam bentuk hash PBKDF2")
    c = conn()
    rows = c.execute("SELECT username,name,role FROM users ORDER BY role,name").fetchall()
    c.close()
    st.dataframe(nice_columns(pd.DataFrame([dict(x) for x in rows])),use_container_width=True,hide_index=True)

# ============================================================
# PANDUAN
# ============================================================
else:
    hero("❓ Panduan Sistem", "Panduan singkat pembagian hak akses dan alur penggunaan aplikasi.", "📖 Gunakan menu sesuai peran")
    role_df = pd.DataFrame([(r,", ".join(v)) for r,v in ROLES.items()],columns=["Peran","Menu yang dapat diakses"])
    st.dataframe(role_df,use_container_width=True,hide_index=True)

    st.markdown('<div class="section-title">🔑 Akun Pengguna</div>',unsafe_allow_html=True)
    users_df = pd.DataFrame(DEFAULT_USERS,columns=["ID Pengguna","Password","Peran","Nama"])
    st.dataframe(users_df,use_container_width=True,hide_index=True)
    st.info("Akun awal disediakan untuk akses pengguna. Gunakan kredensial sesuai peran dan ubah pengaturan keamanan sebelum digunakan dalam lingkungan operasional.")

st.markdown('<div class="footer-note">Prediksi Akademik • Implementasi Algoritma Naive Bayes untuk Prediksi Prestasi Akademik Siswa Berbasis Web</div>',unsafe_allow_html=True)
