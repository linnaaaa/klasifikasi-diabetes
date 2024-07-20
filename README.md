# Aplikasi Klasifikasi Diabetes Menggunakan Decision Tree

Aplikasi ini dibuat untuk mengklasifikasikan apakah seseorang menderita diabetes atau tidak berdasarkan beberapa parameter medis menggunakan algoritma Decision Tree. Aplikasi ini dibangun menggunakan Streamlit.

## Informasi tentang Dataset

Dataset ini berisi informasi tentang pasien dan apakah mereka memiliki diabetes atau tidak. Berikut adalah penjelasan dari setiap kolom dalam dataset:

-   **Pregnancies**: Jumlah kehamilan
-   **Glucose**: Konsentrasi glukosa plasma
-   **BloodPressure**: Tekanan darah diastolik (mm Hg)
-   **SkinThickness**: Ketebalan lipatan kulit triceps (mm)
-   **Insulin**: Kadar insulin serum dua jam (mu U/ml)
-   **BMI**: Indeks Massa Tubuh (berat dalam kg/(tinggi dalam m)^2)
-   **DiabetesPedigreeFunction**: Fungsi silsilah diabetes (skor berdasarkan riwayat diabetes keluarga)
-   **Age**: Usia (tahun)
-   **Outcome**: Hasil (1: Diabetes, 0: Non-Diabetes)

## Cara Menggunakan Aplikasi

### 1. Prasyarat

Pastikan Anda telah menginstal Streamlit dan pustaka Python yang diperlukan. Anda dapat menginstalnya menggunakan pip:

```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn
```

### 2. Menjalankan Aplikasi

Simpan kode aplikasi di atas dalam file bernama diabetes_app.py. Setelah itu, jalankan perintah berikut di terminal Anda:

```bash
streamlit run diabetes_app.py
```

### 3. Menggunakan Aplikasi

Setelah aplikasi berjalan, Anda dapat melihat halaman Streamlit di browser Anda. Ikuti langkah-langkah berikut untuk menggunakan aplikasi:

1. Informasi Dataset: Pada bagian ini, Anda akan melihat informasi tentang dataset dan contoh data.
2. Contoh Input untuk Prediksi dan Hasilnya: Bagian ini menampilkan beberapa contoh input dan hasil prediksi.
3. Buat Prediksi: Masukkan nilai untuk setiap parameter medis (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) dan klik tombol "Prediksi". Aplikasi akan menampilkan hasil prediksi apakah Anda memiliki diabetes atau tidak berdasarkan input yang diberikan.

#### Contoh Input dan hasil

Berikut adalah beberapa contoh input dan hasil prediksinya yang ditampilkan di aplikasi:

-   Contoh 1: [6, 148, 72, 35, 0, 33.6, 0.627, 50] - Hasil: Diabetes
-   Contoh 2: [1, 85, 66, 29, 0, 26.6, 0.351, 31] - Hasil: Non-Diabetes
-   Contoh 3: [8, 183, 64, 0, 0, 23.3, 0.672, 32] - Hasil: Diabetes
-   Contoh 4: [1, 89, 66, 23, 94, 28.1, 0.167, 21] - Hasil: Non-Diabetes
-   Contoh 5: [0, 137, 40, 35, 168, 43.1, 2.288, 33] - Hasil: Diabetes
