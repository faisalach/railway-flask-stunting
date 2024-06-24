from flask import Flask, jsonify
from flask_cors import CORS
import os

from model_food import recommend_foods
from model_stunting import predict_status_gizi

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))