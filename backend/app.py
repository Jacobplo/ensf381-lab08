from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

SEEDED_USERS = {
    "1": {"id": "1", "first_name": "Ava", "user_group": 11},
    "2": {"id": "2", "first_name": "Ben", "user_group": 22},
    "3": {"id": "3", "first_name": "Chloe", "user_group": 33},
    "4": {"id": "4", "first_name": "Diego", "user_group": 44},
    "5": {"id": "5", "first_name": "Ella", "user_group": 55},
}

MODEL_PATH = Path(__file__).resolve().parent / "src" / "random_forest_model.pkl"
PREDICTION_COLUMNS = [
    "city",
    "province",
    "latitude",
    "longitude",
    "lease_term",
    "type",
    "beds",
    "baths",
    "sq_feet",
    "furnishing",
    "smoking",
    "cats",
    "dogs",
]

app = Flask(__name__)
# For this lab, allow cross-origin requests from the React dev server.
# This broad setup keeps local development simple and is not standard
# production practice.
CORS(app)
users = deepcopy(SEEDED_USERS)


@app.route('/users', methods=['GET', 'POST'])
def userEndpoint():
    if request.method == 'GET':
        return jsonify(list(users.values())), 200

    if request.method == 'POST':
        new_user = request.get_json()

        if not new_user or 'id' not in new_user:
            return jsonify({"error": "Invalid input"}), 400

        if new_user['id'] in users:
            return jsonify({"error": "User already exists"}), 409

        users[new_user['id']] = new_user
        return jsonify(list(users.values())), 201


@app.route('/users/<user_id>', methods=['PUT', 'DELETE'])
def userByIdEndpoint(user_id):
    if request.method == 'PUT':
        updated_user = request.get_json()

        if not updated_user:
            return jsonify({"error": "Invalid input"}), 400

        if user_id not in users:
            return jsonify({"error": "User not found"}), 404

        users[user_id] = updated_user
        users[user_id]["id"] = user_id
        return jsonify(users[user_id]), 200

    if request.method == 'DELETE':
        if user_id not in users:
            return jsonify({"error": "User not found"}), 404

        del users[user_id]
        return jsonify({"message": "User deleted"}), 200
#   Exercise2
# - POST /predict_house_price


@app.route("/predict_house_price", methods=["POST"])
def predict_house_price():
  model = joblib.load(MODEL_PATH)
  
  data = request.json

  sample_data = [ data['city'], data['province'], float(data['latitude']),
                 float(data['longitude']), data['lease_term'], data['type'],
                 float(data['beds']), float(data['baths']),
                 float(data['sq_feet']), data['furnishing'], data['smoking'],
                 bool(data['pets']), bool(data['pets']), ]

  sample_df = pd.DataFrame([sample_data], columns=[ 
      'city', 'province', 'latitude', 'longitude', 'lease_term', 
      'type', 'beds', 'baths', 'sq_feet', 'furnishing', 
      'smoking', 'cats', 'dogs' 
  ])

  predicted_price = (model.predict(sample_df))

  return jsonify({"predicted_price": predicted_price}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5050)
