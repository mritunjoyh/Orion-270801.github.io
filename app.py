from flask import Flask, render_template, request, jsonify
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import csv
stemmer = LancasterStemmer()
seat_count = 50

with open("intents.json") as file:
	data = json.load(file)
with open("data.pickle","rb") as f:
	words, labels, training, output = pickle.load(f)

#Function to process input
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w == se:
				bag[i] = 1

	return np.array(bag)

tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

#Loading existing model from disk
from flask_restful import reqparse, abort, Api, Resource
model = tflearn.DNN(net)
model.load("model.tflearn")
app = Flask(__name__)
api = Api(app)
app.static_folder = 'static'
class mj(Resource):
	@app.route('/',methods=['GET'])
	def index():
		return render_template('index.html')
	@app.route('/get',methods = ['GET'])
	def get_bot_response():
		message = request.args.get('msg')
		if message:
			message = message.lower()
			results = model.predict([bag_of_words(message,words)])[0]
			result_index = np.argmax(results)
			tag = labels[result_index]
			for tg in data['intents']:
				if tg['tag'] == tag:
					responses = tg['responses']
			response = random.choice(responses)
			with open("response.csv",'a') as res:
				res = csv.writer(res, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				res.writerow([message,response])
			return str(response)
		return "Missing Data!"
api.add_resource(mj,'/mj')
#api.add_resource(Todo, '/todos/<todo_id>')
if __name__ == "__main__":
	with open("response.csv",'w') as response:
		response = csv.writer(response, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		response.writerow(['User','Bot'])
	app.run()

