import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Memuat dataset dari file CSV
df = pd.read_csv('nutrition.csv')

# Normalisasi atribut
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['calories', 'proteins', 'fat', 'carbohydrate']])
df_normalized = pd.DataFrame(scaled_features, columns=['calories', 'proteins', 'fat', 'carbohydrate'])
df_normalized['name'] = df['name']  # Menambahkan nama makanan ke dataframe hasil normalisasi


# Memisahkan dataset menjadi training dan testing
train_df, test_df = train_test_split(df_normalized, test_size=0.2, random_state=42)

# Menghitung cosine similarity matrix untuk training set
cosine_sim = cosine_similarity(train_df[['calories', 'proteins', 'fat', 'carbohydrate']])

def recommend_foods(input_ids, cosine_sim):
    recommended_food = []

    for id in input_ids:
        index = df[df['id'] == id].index[0]  # Mendapatkan index dari id yang diinput
        sim_scores = list(enumerate(cosine_sim[index]))  # Mendapatkan similarity scores dari makanan yang lain

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Mengurutkan berdasarkan similarity score
        sim_scores = sim_scores[1:3]  # Mengambil 3 makanan teratas yang memiliki similarity tertinggi

        food_indices = [i[0] for i in sim_scores]  # Mengambil index dari makanan yang direkomendasikan
        recommended_food.extend(df['name'].iloc[food_indices].values)  # Menambahkan nama makanan yang direkomendasikan

    return recommended_food

def evaluate_model(input_ids, recommendations):
    liked_food = set([df[df['id'] == id]['name'].values[0] for id in input_ids])
    recommended_food = set(recommendations)

    # Menghitung metrik
    accuracy = len(liked_food) / len(recommended_food) * 100
    precision = len(liked_food) / len(recommended_food) * 100
    recall = len(liked_food) / len(liked_food) * 100
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1

""" # Contoh inputan id makanan yang sering dimakan
input_ids = [2,30, 55,77,29,10,39,59]

# Mendapatkan rekomendasi makanan dari model
recommendations = recommend_foods(input_ids, cosine_sim)

# Evaluasi model
accuracy, precision, recall, f1 = evaluate_model(input_ids, recommendations)

print(recommendations)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
 """