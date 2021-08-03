import nltk
import os
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Conv2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D,GaussianNoise,GlobalMaxPooling1D,UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

#Loading Data
with open("intents.json") as file:
	data = json.load(file)
words = []
labels = []
docs_x = []
docs_y = []

#Looping through our data
for intent in data['intents']:
	for pattern in intent['patterns']:
		pattern = pattern.lower()
    		#Creating a list of words
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent['tag'])

	if intent['tag'] not in labels:
	  labels.append(intent['tag'])
stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x,doc in enumerate(docs_x):
	bag = []
	wrds = [stemmer.stem(w) for w in doc]
	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)
	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1
	training.append(bag)
	output.append(output_row)
#Converting training data into NumPy arrays
training = np.array(training)
output = np.array(output)

#Saving data to disk
with open("data.pickle","wb") as f:
	pickle.dump((words, labels, training, output),f)
tf.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)
model.save("model.tflearn")
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(4,4),input_shape= [None, len(training[0])],activation='relu',padding='same'))
model.add(Dense(256, activation='linear'))
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(4,4),activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))
#model.add(GaussianNoise())
model.add(MaxPooling2D((2,3),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (4,4), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
#model.add(GaussianNoise())
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (4,4), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (4,4), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
#model.add(GaussianNoise())
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
#model.add(Flatten())
model.add(Dropout(0.2))    
#model.add(Dense(256, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Flatten())
model.add(Dense(1, activation='softmax'))
model.fit(training, output, n_epoch = 200, batch_size = 8, show_metric = True)
model.save("model.tflearn")
"""
