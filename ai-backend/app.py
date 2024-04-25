from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)


def get_db_details():
    try:
        client , db, collection = None, None, None
        client = MongoClient('')
        db = client['db_name']
        collection = db['movies']
    except Exception as e:
        print(e)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message":"Hello World"})

if __name__ == '__main__':
    app.run(debug = True)


