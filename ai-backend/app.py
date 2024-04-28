from flask import Flask,Response, request, jsonify, stream_with_context
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from test_sample_model import get_sentiment, apply_training, train_model_stream
import asyncio
import aiohttp

app = Flask(__name__)

client , db, collection = None, None, None
uri = ''
def get_db_details():
    return asyncio.run(get_db_details_async()) 

async def get_db_details_async():
    try:
        client = MongoClient(uri,server_api=ServerApi('1'))
        client.admin.command('ping')
        db = client['sample_mflix']
        collection = db['movies']
        print("Collections:", db.list_collection_names())
        print("Number of documents:", collection.count_documents({}))
        print("Document 1: "+str(collection.find_one()))
    except Exception as e:
        print("Exception while connecting db"+str(e))

@app.route('/hello', methods=['GET'])
def hello():
    print("Getting doc")
    get_db_details()
    print("Returning msg")
    return jsonify({"message":"Hello World"})



@app.route('/sentiment', methods=['POST'])
def predict_sentiment():
    data = request.get_json(force=True)
    review = data['review']
    sentiment = "None"
    sentiment = get_sentiment(review)
    print("Awaiting async response")
    return jsonify({'sentiment':'positive'})

@app.route('/train_model',methods=['GET'])
def applying_training():
    print("Training begins")
    # status = apply_training('sample-demo')
    # return jsonify({'status':status})
    return Response(stream_with_context(train_model_stream()),content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug = True)


