#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries and modules
import io
import pandas as pd
import re
import spacy
import nltk
import numpy as np
import tensorflow as tf
import string
import nltk.tokenize
#NLP packages
from nltk.corpus import stopwords
punc = string.punctuation
stop_words = set(stopwords.words('english'))
#Supervised learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
##Deep learning libraries and APIs
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[3]:


# Load the dataset into a Pandas DataFrame
df = pd.read_csv('DataLatest.csv')


# In[4]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# In[5]:


# Convert Headline and Summary columns to strings and drop any rows with NaN values
df['Headline'] = df['Headline'].astype(str)
df['Summary'] = df['Summary'].astype(str)
df = df.dropna(subset=['Summary'])

# Apply the VADER sentiment analyzer to the Headline column
df['headline_sentiment'] = df['Headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Apply the VADER sentiment analyzer to the Summary column
df['summary_sentiment'] = df['Summary'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Calculate the overall sentiment score as the average of headline and summary sentiment scores
df['sentiment_score'] = (df['headline_sentiment'] + df['summary_sentiment']) / 2


# In[6]:


# Define a function to assign sentiment labels based on the sentiment score
def label_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the function to the sentiment_score column to create the Sentiment column
df['Sentiment'] = df['sentiment_score'].apply(label_sentiment)


# In[7]:


df["text"] = df["Headline"] + " " + df["Summary"]
df.drop('Ticker', inplace=True, axis=1)
df.drop('Headline', inplace=True, axis=1)
df.drop('Datetime', inplace=True, axis=1)
df.drop('Summary', inplace=True, axis=1)
df.drop('Source', inplace=True, axis=1)
df.drop('URL', inplace=True, axis=1)
df.drop('headline_sentiment', inplace=True, axis=1)
df.drop('sentiment_score', inplace=True, axis=1)
df.drop('summary_sentiment', inplace=True, axis=1)
print(df)


# In[8]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

#make a copy of the dataframe
data = df.copy()

# function to perform all pre-processing tasks
def preprocess(text):
    # iterate over each string object in the Series and apply lower() method
    text = text.apply(lambda x: x.lower())
    
    # tokenize text into individual words
    tokens = text.apply(lambda x: word_tokenize(x))
    
    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = tokens.apply(lambda x: [word for word in x if word not in stop_words])
    
    # lemmatize tokens
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = filtered_tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    
    # remove punctuations
    table = str.maketrans('', '', string.punctuation)
    clean_tokens = lemmatized_tokens.apply(lambda x: [word.translate(table) for word in x])
    
    # join the tokens back into a string
    text = clean_tokens.apply(lambda x: ' '.join(x))
    return text

#apply the data preprocessing function
data['text'] = preprocess(data['text'])
data


# In[17]:


from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[25]:


# Split the data into training and testing sets
X = data['text'].values
y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad the tokenized data to have the same length
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Define the deep learning model
model = Sequential()
model.add(Embedding(5000, 128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


# In[26]:


# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
# Define the color map
cmap = plt.get_cmap('Blues')

# Plot the confusion matrix with colors
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[33]:


new_headline = ["Shares of Amazon.com Inc. advanced 2.35% to $104.98 Wednesday, on what proved to be an all-around dismal trading session for the stock market, with the S&P 500 Index falling 0.38% to 4,055.99 and Dow Jones Industrial Average falling 0.68% to 33,301.87"]
##prepare the sequences of the sentences in question
sequences = tokenizer.texts_to_sequences(new_headline)
padded_seqs = pad_sequences(sequences, maxlen=120, padding='post', truncating='post')
print(model.predict(padded_seqs))


# In[ ]:




