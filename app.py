import flask
from flask import Flask, jsonify, request
import analysis
import json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)#, supports_credentials=True)

@app.route('/', methods=["GET", "POST"])
@app.route('/detect', methods=["GET", "POST"])
#@cross_origin(supports_credentials=True)
def detect():
    if request.method == "POST":
        jsonData = json.loads(request.get_data().decode())
        out = analysis.main(jsonData)
        response = jsonify(out)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    return "Hello"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
