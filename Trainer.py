import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open ("intents.json") as file:
    data = json.load(file)

#Load excisting data if it exist, if not will generate above
#then save a file with the name "data.pickle" for future use

words = []
labels = []
docs_y = []
docs_x = []

for intent in data ["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"] # makes all letter low from caps. 
words = sorted(list(set(words))) # mkaes sure to remove all the duplicated 

labels = sorted(labels)
# Create the BagOfWords 
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
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

training = numpy.array(training)
output = numpy.array(output)

    
#below is the structure of the model 
tensorflow.compat.v1.reset_default_graph()
#Find a input shape for our model 
net = tflearn.input_data(shape=[None,len(training[0])])
#2 hiden layers with 8 neurons fully connected to the output layer 
net = tflearn.fully_connected(net,13)
net = tflearn.fully_connected(net,13)
net = tflearn.fully_connected(net,13)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # softmax will give each output a possibility 
net = tflearn.regression(net)

model = tflearn.DNN(net)


#number of epoch is how many time will look through the data, show_metric is for ouputing a graph 
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

