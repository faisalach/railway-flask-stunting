import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Contoh dataset makanan

# Memuat dataset dari file CSV
df = pd.read_csv('nutrition.csv')  

# Normalisasi atribut
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['calories', 'proteins', 'fat', 'carbohydrate']])
df_normalized = pd.DataFrame(scaled_features, columns=['calories', 'proteins', 'fat', 'carbohydrate'])
df_normalized['name'] = df['name']  # Menambahkan nama makanan ke dataframe hasil normalisasi


# Menghitung cosine similarity matrix
cosine_sim = cosine_similarity(df_normalized[['calories', 'proteins', 'fat', 'carbohydrate']])

def recommend_foods(input_ids):
    recommended_food = []

    for id in input_ids:
        index = df[df['id'] == id].index[0]  # Mendapatkan index dari id yang diinput
        sim_scores = list(enumerate(cosine_sim[index]))  # Mendapatkan similarity scores dari makanan yang lain

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Mengurutkan berdasarkan similarity score
        sim_scores = sim_scores[1:4]  # Mengambil 3 makanan teratas yang memiliki similarity tertinggi

        food_indices = [i[0] for i in sim_scores]  # Mengambil index dari makanan yang direkomendasikan
        recommended_food.extend(df['name'].iloc[food_indices].values)  # Menambahkan nama makanan yang direkomendasikan

    return recommended_food

# Contoh inputan id makanan yang sering dimakan
""" input_ids = [2, 30]

# Mendapatkan rekomendasi makanan
recommendations = recommend_foods(input_ids)
print("Rekomendasi Makanan:")
for food in recommendations:
    print(food) """

def calculate_accuracy(input_ids, recommendations):
    liked_food = set([df[df['id'] == id]['name'].values[0] for id in input_ids])
    recommended_food = set(recommendations)

    intersection = liked_food.intersection(recommended_food)
    accuracy = len(intersection) / len(recommended_food) * 100

    return accuracy

# Menghitung presentase keakuratan
""" accuracy = calculate_accuracy(input_ids, recommendations)
print(f"Presentase keakuratan: {accuracy:.2f}%") """
