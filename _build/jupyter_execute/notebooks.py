#!/usr/bin/env python
# coding: utf-8

# ### [Линк](https://colab.research.google.com/drive/1mguVQuMEn2mIfISPCf4I9P6rvjNAK2ub?usp=sharing) до самиот Notebook

# ### Вовед

# #### Import на библиотеките кои се користат во кодот

# In[1]:


from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report, f1_score, log_loss, precision_score, recall_score

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)


# #### Симнување на glove векторите за репрезентација на зборови

# In[2]:


get_ipython().system('wget http://nlp.stanford.edu/data/glove.6B.zip')
get_ipython().system('unzip glove*.zip')


# ### Помошни функции

# #### Исчитување на glove фајлот со вредности

# In[3]:


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
            
    return words_to_index, index_to_words, word_to_vec_map


# #### Softmax функцијата

# In[4]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    
    return e_x / e_x.sum()


# #### Читање на соодветните вредности од CSV train и test датотеките

# In[5]:


def read_csv(filename):
    phrase = []
    emoji_ = []

    with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)

        for row in csv_reader:
            phrase.append(row[0])
            emoji_.append(row[1])

    x = np.asarray(phrase)
    y = np.asarray(emoji_, dtype=int)

    return x, y


# #### Излезите(бројки) ги претвора во one-hot вектори

# In[6]:


def convert_to_one_hot(y, c):
    y = np.eye(c)[y.reshape(-1)]
    
    return y


# #### Предвидување на излезите при дадени елементи како влез

# In[7]:


def predict(X, Y, W, b, word_to_vec_map):
    m = X.shape[0]
    pred = np.zeros((m, 1))

    for j in range(m):
        words = X[j].lower().split()

        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg / len(words)

        z = np.dot(W, avg) + b
        a = softmax(z)
        pred[j] = np.argmax(a)

    print("Accuracy: " + str(np.mean((pred[:] == Y.reshape(Y.shape[0], 1)[:]))))

    return pred


# #### Претворање на реченици дадени како влез во матрици од вредности

# In[8]:


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0] 
    x_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = (X[i].lower()).split()
        j = 0
    
        for w in sentence_words:
            # i-тата вредност е редниот број на реченицата, j-тата е редниот
            # број на зборот во неа. вредноста која се поставува е таа на 
            # соодветниот збор
            x_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return x_indices


# #### Креирање на веќе истрениран Embedding слој со помош на glove векторите

# In[9]:


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# #### Креирање на модел со соодветни предефинирани вредности

# In[10]:


def sentiment_analysis(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)

    # LSTM слој со 128-димензионален hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    
    # Веројатност на зачувување од 0.5
    X = Dropout(0.5)(X)
    
    # Уште еден LSTM слој со 128-димензионален hidden state
    X = LSTM(128)(X)

    X = Dropout(0.5)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)

    model = Model(sentence_indices, X)

    return model


# ### Main дел

# #### Читање на train и test податоците, пренос на излезите како one-hot вектори, читање на векторите за репрезентација на зборови

# In[11]:


X_train, Y_train = read_csv('datasets/train_set.csv')
X_test, Y_test = read_csv('datasets/test_set.csv')

maxLen = len(max(X_train, key=len).split())

Y_oh_train = convert_to_one_hot(Y_train, 5)
Y_oh_test = convert_to_one_hot(Y_test, 5)

word_to_index_, index_to_word, word_to_vec_map_ = read_glove_vecs('glove.6B.50d.txt')


# #### Креирање и тренирање на моделот

# In[12]:


model = sentiment_analysis((maxLen,), word_to_vec_map_, word_to_index_)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index_, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, 5)

model.fit(X_train_indices, Y_train_oh, epochs=100, batch_size=32, shuffle=True)


# #### Тестирање на моделот со test податоците

# In[13]:


X_test_indices = sentences_to_indices(X_test, word_to_index_, max_len=maxLen)
Y_test_oh = convert_to_one_hot(Y_test, 5)

loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Accuracy ", acc)


# #### Преглед на влезовите кои се грешно предвидени и дополнителни метрики за евалуација

# In[14]:


y_test_oh = np.eye(5)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index_, maxLen)
pred = model.predict(X_test_indices)

actual = []
predicted = []

for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])

    actual.append(Y_test[i])
    predicted.append(num)

    if num != Y_test[i]:
        print('Input: ' + str(X_test[i]))
        print('Expected class: ' + str(Y_test[i]))
        print('Predicted class: ' + str(num) + '\n')

precision = precision_score(actual, predicted, average='macro')
recall = recall_score(actual, predicted, average='macro')
f1_score = f1_score(actual, predicted, average='macro')
loss = log_loss(actual, pred, eps=1e-15)
matrix = classification_report(actual, predicted, labels=[0, 1, 2, 3, 4])


# #### Тестирање на моделот со влезови од корисник

# In[15]:


x_test = np.array(['very happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index_, maxLen)
print('Input: ' + x_test[0])
print('Predicted class: ' + str(np.argmax(model.predict(X_test_indices))) + '\n')


# In[16]:


x_test = np.array(['very sad'])
X_test_indices = sentences_to_indices(x_test, word_to_index_, maxLen)
print('Input: ' + x_test[0])
print('Predicted class: ' + str(np.argmax(model.predict(X_test_indices))) + '\n')


# In[17]:


x_test = np.array(['i am starving'])
X_test_indices = sentences_to_indices(x_test, word_to_index_, maxLen)
print('Input: ' + x_test[0])
print('Predicted class: ' + str(np.argmax(model.predict(X_test_indices))) + '\n')


# In[18]:


x_test = np.array(['I have met the love of my life'])
X_test_indices = sentences_to_indices(x_test, word_to_index_, maxLen)
print('Input: ' + x_test[0])
print('Predicted class: ' + str(np.argmax(model.predict(X_test_indices))) + '\n')


# #### Приказ на мерките за успешност на моделот

# In[19]:


print('Accuracy: {0}'.format(str(acc)))


# In[20]:


print('Log loss: {0}'.format(loss))


# In[21]:


print('Precision: {0}'.format(precision))


# In[22]:


print('Recall: {0}'.format(recall))


# In[23]:


print('F1 score: {0}'.format(f1_score))


# In[24]:


print('Classification report: \n{0}'.format(matrix))

