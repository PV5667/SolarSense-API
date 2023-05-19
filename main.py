import flask
from flask import Flask, jsonify, request
from analysis import main
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET", "POST"])
@app.route('/detect', methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        jsonData = json.loads(request.get_data().decode())
        out = main(jsonData)
        response = jsonify(out)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    return "Hello"
