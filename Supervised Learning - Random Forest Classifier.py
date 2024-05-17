#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[57]:


# Load the dataset into a Pandas DataFrame
df = pd.read_csv('DataLatest.csv')


# In[58]:


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


# In[59]:


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


# In[60]:


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


# In[61]:


headline_counts = df['Sentiment'].value_counts()
print(headline_counts)


# In[62]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer


# In[63]:


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


# In[64]:


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


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Define your feature matrix and target variable
tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(data['text'])
y = data['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[66]:


from sklearn.metrics import confusion_matrix

# Make predictions on test set using trained classifier
y_pred = rf.predict(X_test)

# Get confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print(cm)


# In[67]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[68]:


new_data = ["The US imposes sanctions on Rassia because of the Ukranian war"]
tf = TfidfVectorizer()
tfdf = tf.fit_transform(data['text'])
vect = pd.DataFrame(tf.transform(new_data).toarray())
new_data = pd.DataFrame(vect)
logistic_prediction = rf.predict(new_data)
print(logistic_prediction)


# In[ ]:




