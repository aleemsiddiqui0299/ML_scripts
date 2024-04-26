from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import asyncio
import aiohttp

app = Flask(__name__)

client , db, collection = None, None, None

async def get_db_details():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_db_details_async)

async def get_db_details_async():
    try:
        client = MongoClient('')
        db = client['samle_mflix']
        collection = db['movies']
        print("Document 1: "+str(collection.find_one()))
    except Exception as e:
        print("Exception while connecting db"+str(e))

@app.route('/hello', methods=['GET'])
def hello():
    print("Getting doc")
    get_db_details()
    print("Returning msg")
    return jsonify({"message":"Hello World"})

if __name__ == '__main__':
    app.run(debug = True)


