import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents2.json').read())
words = []
classes = []
documents = []
ignoreLetters = ['?','!','.',',']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
nltk.download('wordnet')
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]     
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
training = []
outputEmpty = [0] * len(classes)   
for document in documents:
    bag = []
    wordPatterens = document[0]
    wordPatterens = [lemmatizer.lemmatize(word.lower()) for word in wordPatterens]
    for word in words:
        bag.append(1) if word in wordPatterens else bag.append(0)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)  
random.shuffle(training)
training = np.array(training)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape = (len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation ='softmax'))
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01 , momentum=0.9, nesterov = True )
model.compile(loss ='categorical_crossentropy', optimizer= sgd , metrics=['accuracy'])
model.summary()
#model.fit(trainX,trainY,epochs = 300,batch_size=10, verbose= 1)
hist = model.fit(np.array(trainX),np.array(trainY),epochs = 300 ,batch_size = 10 , verbose = 1)
model.save('chat_model.h5',hist)
print('Done')







