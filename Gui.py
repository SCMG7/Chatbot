from tkinter import *
import tkinter as tkr
import nltk 
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
import json
import pickle
import tflearn
stemmer = LancasterStemmer()





with open ("intents.json") as file:
	data = json.load(file)

#Load excisting data if it exist, if not will generate above
#then save a file with the name "data.pickle" for future use
try:
	#Save all the varible in pickle file-this list is the only list we need for our model 
	#Delete pickle file in case intents are modife
	with open("data.pickle", "rb") as f:# rb stands for read bites
		words, labels , training, output = pickle.load(f)
except:
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

	with open("data.pickle", "wb") as f:# wb stands for write bites
		pickle.dump((words, labels , training, output), f) 
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

try:
	model.load("model.tflearn")
except:
	#number of epoch is how many time will look through the data, show_metric is for ouputing a graph 
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s, words):
	bag=[0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	#Generating the bag or words
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

root = tkr.Tk()
root.geometry("1200x538")
root.title('Marti Georgiev')
root['background']='#000000'
root.resizable(False, False)


#Visuals Below and graphical user interface stracute 
#Create a menu bar
main_menu = Menu(root)
#Sub-menu
file_menu = Menu(root)
#file_menu['background']='#36cbd9'
file_menu.add_command(label='Export conversation')
file_menu.add_command(label='Upload a file')
file_menu.add_command(label='New conversation')
#main menu
main_menu.add_cascade(label='File', menu = file_menu)
main_menu.add_command(label='Report')
main_menu.add_command(label='Exit')
root.config(menu=main_menu)
#ChatWindow 

#Message Window
msgWindow_entry =Entry(root, bg='#656565', fg='black', font='bold')
msgWindow_entry.place(x=0,y=480, height=40, width=1110)


#Ask the user for input and give answer code below,
chatWindow = Label(root,text="Start talking with the bot!  ((Type exit to stop))",bg='#424242', fg='black' ,width=90, height = 9)
chatWindow.place(x=0,y=0, height=480, width=1200)
def chat():
	#print("Start talking with the bot!  ((Type exit to stop))")
	#print("Start talking with the bot!  ((Type exit to stop))")
	while True:
		
		inp = input("You: ")
		#Save the history of inputs in a txt file called ChatHistory
		with open('ChatHistory.txt', 'a') as rf:
			rf.write(inp + '\n')
		

		if inp.lower() == "exit":
			break
		results = model.predict([bag_of_words(inp, words)])[0]
		#will find the max possibility on a 'tag' and use that one.
		results_index = numpy.argmax(results)
		#This will give us the name of the tag or the label of the tag 
		tag = labels[results_index]
		#print(tag)
		# Pick a answer from the selected tag.
		#if index is more than 70% will display a selected masssage
		if results[results_index] > 0.7:
			for tg in data["intents"]:
				if tg['tag'] ==tag:
					responses = tg['responses']
			rcrs = (random.choice(responses))
			chatWindow = Label(root,text=rcrs, bg='#424242', fg='black' ,width=90, height = 9)
			chatWindow.place(x=0,y=0, height=480, width=1200)
			#print(random.choice(responses))
		else:
			chatWindow = Label(root,text="I could not understand that. Please try again.", bg='#424242', fg='black' ,width=90, height = 9)
			chatWindow.place(x=0,y=0, height=480, width=1200)
			#print("I could not understand that. Please try again.")
			#print("I could not understand that. Please try again.")

def clean ():
	msgWindow_entry.delete(0, END)

#Button send 
btn_submit = Button(root,text="SEND",height=2, width=12,command=clean)
btn_submit.place(x=1110,y=480)

auth_lbl=Label(root,text="Graphical User Interface created by Marti Georgiev",fg='green',bg="black")
auth_lbl.config(font=("Courier", 8))
auth_lbl.place(x=420,y=520)
chat()
tkr.mainloop()

