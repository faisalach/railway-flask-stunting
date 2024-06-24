from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from model_food import recommend_foods
from model_stunting import predict_status_gizi

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

@app.route("/api/foods",methods=['POST'])
def food():
    food_ids    = request.json['food_ids']
    umur    = request.json['umur']
    jenis_kelamin    = request.json['jenis_kelamin']
    tinggi_badan    = request.json['tinggi_badan']

    predict = predict_status_gizi(umur, jenis_kelamin, tinggi_badan)
    result  = recommend_foods(food_ids)
    return jsonify({
        'food' : result,
        'stunting': {
            'stunting': predict,
            'umur': umur,
            'jenis_kelamin': jenis_kelamin,
            'tinggi_badan': tinggi_badan
        }})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))