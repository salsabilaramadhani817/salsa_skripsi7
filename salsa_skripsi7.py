#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('pip', 'install statsmodels')


# In[3]:


import statsmodels.api as sm
print("statsmodels berhasil diinstal!")


# In[4]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import statsmodels.api as sm

print("Python yang diminta: 3.13.7")
print("JupyterLab yang diminta: 4.4.7")
print("Library berhasil dipanggil.")


# In[8]:


data = pd.read_csv(r"C:\Users\ASUS\Documents\SKRIPSI\supply_chain_data.csv")

print("Ukuran data:", data.shape)
display(data.head())
print("\\nNama kolom:")
print(data.columns.tolist())


# In[9]:


display(data.info())
display(data.describe(include="all").T)
print("Jumlah data kosong per kolom:")
display(data.isnull().sum())


# In[10]:


# Cari kolom Shipping times tanpa sensitif huruf besar/kecil
kolom_map = {c.strip().lower(): c for c in data.columns}
shipping_col = kolom_map.get("shipping times")

if shipping_col is None:
    raise ValueError(
        "Kolom 'Shipping times' tidak ditemukan. Periksa nama kolom dengan print(data.columns.tolist()), "
        "lalu sesuaikan kode target."
    )

# Target proxy: 1 jika waktu pengiriman di atas median, 0 jika sama dengan/di bawah median
median_shipping = data[shipping_col].median()
data["Risiko_Keterlambatan"] = (data[shipping_col] > median_shipping).astype(int)

print("Median Shipping times:", median_shipping)
print("Distribusi label (0 = waktu tidak di atas median, 1 = di atas median):")
display(data["Risiko_Keterlambatan"].value_counts().sort_index())


# In[11]:


target_col = "Risiko_Keterlambatan"

# Ambil kolom numerik selain target dan kolom yang dipakai membentuk target
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
excluded_cols = {target_col, shipping_col}

# Kolom identitas/kode tidak cocok diperlakukan sebagai angka kuantitatif
for c in data.columns:
    if c.strip().lower() in {"sku", "product type"}:
        excluded_cols.add(c)

feature_cols = [c for c in numeric_cols if c not in excluded_cols]

if len(feature_cols) == 0:
    raise ValueError("Tidak ada fitur numerik yang bisa digunakan. Periksa struktur dataset.")

X = data[feature_cols].copy()
y = data[target_col].copy()

print("Variabel prediktor:")
print(feature_cols)
print("\\nJumlah kelas:")
print(y.value_counts())


# In[12]:


# Stratifikasi menjaga proporsi kelas di training dan testing semirip mungkin
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Jumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])


# In[13]:


knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Pilih nilai K yang tidak melebihi jumlah data pada tiap lipatan training
jumlah_fold = min(5, int(y_train.value_counts().min()))
if jumlah_fold < 2:
    raise ValueError("Jumlah data pada salah satu kelas terlalu sedikit untuk validasi silang.")

cv = StratifiedKFold(n_splits=jumlah_fold, shuffle=True, random_state=42)
k_max = max(1, min(15, len(X_train) - 1))
param_grid = {"knn__n_neighbors": list(range(1, k_max + 1, 2))}

grid_knn = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)
grid_knn.fit(X_train, y_train)

model_knn = grid_knn.best_estimator_
pred_knn = model_knn.predict(X_test)

print("Nilai K terbaik:", grid_knn.best_params_["knn__n_neighbors"])
print("Skor F1 validasi silang terbaik:", round(grid_knn.best_score_, 4))


# In[15]:


def fit_logit_with_pvalue_selection(Xtr, ytr, alpha=0.05):
    selected = list(Xtr.columns)
    history = []

    while len(selected) > 1:
        X_model = sm.add_constant(Xtr[selected], has_constant="add")

        try:
            result = sm.Logit(ytr, X_model).fit(disp=False, maxiter=500)
            pvalues = result.pvalues.drop(labels=["const"], errors="ignore")

            if pvalues.empty:
                break

            worst_feature = pvalues.idxmax()
            worst_p = float(pvalues[worst_feature])

            if worst_p <= alpha:
                return result, selected, history

            history.append((worst_feature, worst_p))
            selected.remove(worst_feature)

        except Exception:
            # Hentikan seleksi jika model gagal konvergen
            break

    # Tetap pasang model dengan variabel yang tersisa
    X_model = sm.add_constant(Xtr[selected], has_constant="add")
    result = sm.Logit(ytr, X_model).fit(disp=False, maxiter=500)

    return result, selected, history


hasil_logit, selected_features, riwayat_p = fit_logit_with_pvalue_selection(
    X_train_scaled, y_train, alpha=0.05
)

print("Variabel yang dipertahankan:", selected_features)
print("\nRiwayat penghapusan variabel:")
display(pd.DataFrame(
    riwayat_p,
    columns=["Variabel", "p-value saat dikeluarkan"]
))
print("\nRingkasan model akhir:")
print(hasil_logit.summary())


# In[16]:


X_test_logit = sm.add_constant(
    X_test_scaled[selected_features],
    has_constant="add"
)

prob_lr = hasil_logit.predict(X_test_logit)
pred_lr = (prob_lr >= 0.5).astype(int)

print("Contoh probabilitas prediksi:")
display(pd.DataFrame({
    "Label_asli": y_test,
    "Probabilitas_kelas_1": prob_lr,
    "Prediksi_Logistik": pred_lr
}).head())


# In[17]:


def hitung_metrik(nama, y_true, y_pred):
    return {
        "Model": nama,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }

hasil_evaluasi = pd.DataFrame([
    hitung_metrik("KNN", y_test, pred_knn),
    hitung_metrik("Regresi Logistik Biner", y_test, pred_lr)
]).set_index("Model")

display(hasil_evaluasi.round(4))

print("Confusion matrix KNN:")
print(confusion_matrix(y_test, pred_knn))

print("\\nConfusion matrix Regresi Logistik Biner:")
print(confusion_matrix(y_test, pred_lr))


# In[18]:


display(hasil_evaluasi.round(4).T)

for metrik in ["Accuracy", "Precision", "Recall", "F1-Score"]:
    nilai_knn = hasil_evaluasi.loc["KNN", metrik]
    nilai_lr = hasil_evaluasi.loc["Regresi Logistik Biner", metrik]

    if np.isclose(nilai_knn, nilai_lr):
        pemenang = "memiliki nilai yang sama"
    elif nilai_knn > nilai_lr:
        pemenang = "lebih tinggi pada KNN"
    else:
        pemenang = "lebih tinggi pada Regresi Logistik Biner"

    print(f"{metrik}: {pemenang} (KNN={nilai_knn:.3f}; Regresi Logistik={nilai_lr:.3f})")


# In[ ]:





# In[19]:


hasil_evaluasi.to_csv("hasil_perbandingan_knn_logistik.csv")
print("File tersimpan: hasil_perbandingan_knn_logistik.csv")


# In[ ]:




