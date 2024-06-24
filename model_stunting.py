import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



# Contoh dataset sederhana (gantilah dengan dataset yang sesuai)
""" data = {
    'Umur': [12, 18, 24, 30, 36, 42, 48, 54, 60],
    'Jenis Kelamin': [0, 1, 0, 1, 0, 1, 0, 1, 0],  # 0: Perempuan, 1: Laki-laki
    'Tinggi Badan': [75, 80, 82, 85, 86, 88, 89, 90, 92],
    'Status Gizi': ['normal', 'stunted', 'stunted', 'normal', 'severely stunted', 'stunted', 'normal', 'stunted', 'severely stunted']
}

df = pd.DataFrame(data) """

df = pd.read_csv('data_balita.csv') 
# print(df)

# Ubah kategori menjadi numerik (gunakan LabelEncoder atau sesuaikan dengan kebutuhan)
# Misalnya, untuk Status Gizi, normal=0, stunted=1, severely stunted=2

# Convert categorical variables to numerical using LabelEncoder
label_encoder = LabelEncoder()
df['Jenis Kelamin'] = label_encoder.fit_transform(df['Jenis Kelamin'])

# Define the label encoding mappings (for reference)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
""" print("Label Encoding Mapping:")
print(label_mapping) """

# Pisahkan fitur dan label
X = df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
y = df['Status Gizi']

# Split dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
""" print(f'Akurasi model: {accuracy * 100:.2f}%') """

def predict_status_gizi(umur, jenis_kelamin, tinggi_badan):
    # Prediksi menggunakan model yang sudah dilatih
    input_data = np.array([[umur, jenis_kelamin, tinggi_badan]])
    prediction = model.predict(input_data)
    return prediction[0]
""" 
predicted_status = predict_status_gizi(3, 0, 20)
print(predicted_status)
 """
