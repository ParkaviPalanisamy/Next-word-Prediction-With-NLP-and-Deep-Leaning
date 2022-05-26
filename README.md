# Next-word-Prediction-With-NLP-and-Deep-Leaning
# Import the required libraries
import nltk

import re

import numpy as np

import tensorflow

import keras

from keras.models import Sequential, load_model

import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation

from tensorflow.keras.preprocessing.text import Tokenizer

import pickle

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam

import numpy as np

import os

import pydot

import pydotplus

from pydotplus import graphviz

from keras.utils.vis_utils import plot_model

from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pydot


# Read the dataset 
path=open("C:\\Users\\LENOVO\\Downloads\\text.txt","r",encoding='utf-8')

text =path.read().lower()

print(text[:200])

print("length of text is:",len(text))


# Cleaning the text
sentance=nltk.sent_tokenize(text)

print(sentance)

corpus=[]
for i in range(len(sentance)):
    sent=re.sub('[^a-zA-z]', " ",sentance[i])
    corpus.append(sent)
corpus="".join(corpus)    

corpus=[]

for i in range(len(sentance)):

	sent=re.sub('[^a-zA-z]', " ",sentance[i])
    
	corpus.append(sent)

corpus="".join(corpus)    

print(corpus)

type(corpus)


word=nltk.word_tokenize(corpus)

print(word)

# Using keras tokenizer
tokenizer = Tokenizer()

tokenizer.fit_on_texts([word])

# saving the tokenizer for predict function

pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([word])[0]

sequence_data[:15]

len(sequence_data)
# Getting unique words
vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)
# Feature Engineering
sequences = []

for i in range(1, len(sequence_data)):
    
	words = sequence_data[i-1:i+1]
    
	sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))

sequences = np.array(sequences)

sequences[:10]

# Storing features and labels
X = []

y = []

for i in sequences:
    
	X.append(i[0:1])
    
	y.append(i[1])
    
X = np.array(X)

y = np.array(y)

print("Data: ", X[:10])

print("Response: ", y[:10])

y = to_categorical(y, num_classes=vocab_size)

y[:5]

# Creating the Model
model = Sequential()

model.add(Embedding(vocab_size, 10, input_length=3))

model.add(LSTM(1000, return_sequences=True))

model.add(LSTM(1000))

model.add(Dense(1000, activation="relu"))

model.add(Dense(vocab_size, activation="softmax"))

model.summary()
#to get a model plot

from tensorflow import keras

from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)

# Training and saving the model

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))

model.fit(X, y, epochs=10, batch_size=128, callbacks=[checkpoint])

# Testing next word
## Load the model and tokenizer
model = load_model('next_words.h5')

tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

  
  sequence = tokenizer.texts_to_sequences([text])
  
  sequence = np.array(sequence)
  
  preds = np.argmax(model.predict(sequence))
  
  predicted_word = ""   

  
  for key, value in tokenizer.word_index.items():
      
	  if value == preds:
          
		  predicted_word = key
          
		  break
  
  print(predicted_word)
  
  return predicted_word
  
  while(True):
  
  text = input("Enter your line: ")
  
  if text == "0":
      
	  print("Execution completed.....")
      
	  break
  
  else:
      
	  try:
          
		  text = text.split(" ")
          
		  text = text[-1:]
          
		  print(text)
        
          Predict_Next_Words(model, tokenizer, text)
          
      except Exception as e:
        
		print("Error occurred: ",e)
        
		continue
