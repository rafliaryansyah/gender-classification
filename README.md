# Klasifikasi Jenis Kelamin dari Nama Bahasa Indonesia

Proyek Machine Learning untuk mengklasifikasikan jenis kelamin berdasarkan nama orang Indonesia menggunakan algoritma Logistic Regression, Naive Bayes, dan Random Forest.

## 📊 Dataset

Dataset berasal dari data pemilih tetap KPU yang berisi 13.137 nama dengan label jenis kelamin. Dataset tersedia di [`data/data-pemilih-kpu.csv`](./data/data-pemilih-kpu.csv).

| Nama | Jenis Kelamin |
|------|---------------|
|ERWIN TJAHJONO|Laki-Laki|
|AYU DWI CAHYANING MUKTI|Perempuan|

## 🎯 Metode Klasifikasi

- **Logistic Regression** - Akurasi: ~93.6%
- **Naive Bayes** - Akurasi: ~93.3%
- **Random Forest** - Akurasi: ~93.2%

## 🚀 Quick Start

```bash
# Clone repository
git clone <repository-url>
cd Klasifikasi-jenis-kelamin-dari-nama-bahasa-indonesia-menggunakan-Machine-Learning

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Run prediction
python jenis-kelamin.py "Nama Anda" -ml LG
```

## 📝 Contoh Penggunaan

```bash
# Menggunakan Logistic Regression
python jenis-kelamin.py "Siti Nurhaliza" -ml LG

# Menggunakan Naive Bayes
python jenis-kelamin.py "Ahmad Rizki" -ml NB

# Training ulang dengan dataset baru
python jenis-kelamin.py "Nama" -ml LG -t "./data/dataset-baru.csv"
```

## 📁 Struktur Project

```
├── data/
│   ├── data-pemilih-kpu.csv    # Dataset
│   ├── pipe_lg.pkl              # Model Logistic Regression
│   ├── pipe_nb.pkl              # Model Naive Bayes
│   └── pipe_rf.pkl              # Model Random Forest
├── visualizations/              # Grafik hasil klasifikasi
├── Preksi-KL.ipynb             # Jupyter notebook
├── jenis-kelamin.py            # Script utama
├── LAPORAN.md                  # Laporan lengkap
└── requirements.txt            # Dependencies
```

## 📚 Dokumentasi Lengkap

Untuk laporan lengkap dengan analisis detail, visualisasi, dan hasil eksperimen, lihat [LAPORAN.md](./LAPORAN.md).
