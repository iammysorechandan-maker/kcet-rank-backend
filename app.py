from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import RankPredictor

app = Flask(__name__)
CORS(app)

predictor = RankPredictor("data.csv")

@app.route("/")
def home():
    return "KCET Backend Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    aggregate = float(data["aggregate"])

    result = predictor.predict(aggregate)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)