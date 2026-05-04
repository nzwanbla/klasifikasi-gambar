# Klasifikasi Gambar Pemandangan Menggunakan Transfer Learning MobileNetV2

## Deskripsi Proyek
Proyek ini merupakan implementasi klasifikasi gambar pemandangan menggunakan metode Transfer Learning dengan arsitektur MobileNetV2. Dataset yang digunakan adalah **Intel Image Classification** yang berisi gambar pemandangan alam dan perkotaan dalam 6 kategori kelas.

## Dataset
- **Nama Dataset**: Intel Image Classification
- **Sumber**: [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Jumlah Gambar**: ±14.000 gambar (train) + ±3.000 gambar (test)
- **Total**: ±17.000 gambar
- **Resolusi**: Tidak seragam (beragam ukuran)

## Kelas
Dataset terdiri dari 6 kelas:
| No | Kelas | Deskripsi |
|----|-------|-----------|
| 1 | buildings | Gedung / Bangunan |
| 2 | forest | Hutan |
| 3 | glacier | Gletser |
| 4 | mountain | Gunung |
| 5 | sea | Laut |
| 6 | street | Jalanan |

## Pembagian Dataset
| Set | Proporsi | Jumlah |
|-----|----------|--------|
| Train | 70% | ±9.800 gambar |
| Validation | 20% | ±2.800 gambar |
| Test | 10% | ±1.400 gambar |

## Arsitektur Model
Model menggunakan **Transfer Learning** dengan base model **MobileNetV2** yang telah dilatih pada dataset ImageNet.

### Struktur Model:
- **Base Model**: MobileNetV2 (pretrained ImageNet, input 224x224x3)
- **GlobalAveragePooling2D**
- **BatchNormalization**
- **Dense(256, activation='relu')**
- **Dropout(0.5)**
- **Dense(128, activation='relu')**
- **Dropout(0.3)**
- **Dense(6, activation='softmax')**

### Fine-tuning:
Setelah training awal dengan base model ter-freeze, dilakukan fine-tuning dengan membuka 50 layer terakhir MobileNetV2 menggunakan learning rate yang sangat kecil (0.00001).

## Callback yang Digunakan
1. **ModelCheckpoint** - Menyimpan model terbaik secara otomatis
2. **EarlyStopping** - Menghentikan training jika tidak ada kemajuan (patience=7)
3. **ReduceLROnPlateau** - Mengurangi learning rate saat training stagnan
4. **AccuracyThresholdCallback** - Menghentikan training saat target akurasi tercapai

## Hasil Training
| Metrik | Nilai |
|--------|-------|
| Akurasi Training | ≥ 95% |
| Akurasi Validasi | ≥ 93% |
| Akurasi Test | ≥ 93% |

## Format Model yang Disimpan
| Format | Lokasi | Kegunaan |
|--------|--------|----------|
| SavedModel | `saved_model/` | Deployment server/cloud |
| TF-Lite | `tflite/model.tflite` | Perangkat mobile/embedded |
| TensorFlow.js | `tfjs_model/` | Aplikasi berbasis browser |

## Struktur Direktori
```
klasifikasi-gambar/
├── tfjs_model/
│   ├── group1-shard1of4.bin
│   ├── group1-shard2of4.bin
│   ├── group1-shard3of4.bin
│   ├── group1-shard4of4.bin
│   └── model.json
├── tflite/
│   ├── model.tflite
│   └── label.txt
├── saved_model/
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Requirements
Lihat file `requirements.txt` untuk daftar lengkap library yang digunakan.

## Cara Menjalankan
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Buka file `notebook.ipynb` menggunakan Jupyter Notebook atau Google Colab
3. Jalankan setiap cell secara berurutan

## Inference
Contoh inference menggunakan TF-Lite:
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
interpreter = tf.lite.Interpreter(model_path='tflite/model.tflite')
interpreter.allocate_tensors()

# Load dan preprocess gambar
img = Image.open('gambar.jpg').convert('RGB').resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Prediksi
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Hasil
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
pred_class = classes[np.argmax(output[0])]
confidence = np.max(output[0]) * 100
print(f'Prediksi: {pred_class} ({confidence:.2f}%)')
```

## Author
Nazwa Nabila
