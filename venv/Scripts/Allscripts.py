import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
#===================================================================================
#!pip install openpyxl
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# ==================Increasing the maximum field size limit for python engine==================================
import csv
import sys
csv.field_size_limit(2147483647)
data = pd.read_csv('train.csv',engine='python',on_bad_lines='warn')
data.dropna(inplace=True)
features = pd.DataFrame(data[['title','author','text']])
target = data['label']
#==========================pre processing===================================
#'==============================================PRE-PROCESSING OF TITLE FEATURE==========================================
datalistTit = list(features["title"])
datalistTit = [i.lower() for i in datalistTit] #LOWER
removepunchTit = [re.sub("[^a-zA-Z\s]",'',str(word)) for word in datalistTit] #REMOVE PUNCTUATIONS
removepunchTit = [i.strip() for i in removepunchTit] #REMOVE SPACES
datatokenisedTit = [word_tokenize(i) for i in removepunchTit] #TOKENIZE
datatokenisedTit[1]
# STOP WORDS
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
stop_words.remove("don't")
stop_words.remove("won't")
stop_words.remove("wouldn't")
stop_words.remove("shouldn't")
stop_words.remove("no")
stop_words.remove("nor")
removedstopTit = [[i for i in j if i not in stop_words] for j in datatokenisedTit]
# STEMMING
Porter = PorterStemmer()
stemmedTit = [[Porter.stem(i) for i in j] for j in removedstopTit]
#FLATEN WORDS
flatenedwordsTit = [i for j in stemmedTit for i in j]
cleaned_Title = [" ".join(i) for i in stemmedTit]
#'==========================================PRE PROCESSING AND CLEANING COMPLETED FOR TITLE FEATURE========================
#'==========================================PRE-PROCESSING OF AUTHOR FEATURE========================
datalistauth = list(features["author"])
datalistauth = [i.lower() for i in datalistauth] #LOWER
removepunchauth = [re.sub("[^a-zA-Z\s]",'',str(word)) for word in datalistauth] #REMOVE PUNCTUATIONS
removepunchauth = [i.strip() for i in removepunchauth] #REMOVE SPACES
datatokenisedauth = [word_tokenize(i) for i in removepunchauth] #TOKENIZE
datatokenisedauth[1]
# STOP WORDS
removedstopauth = [[i for i in j if i not in stop_words] for j in datatokenisedauth]
# STEMMING
Porter = PorterStemmer()
stemmedauth = [[Porter.stem(i) for i in j] for j in removedstopauth]
#FLATEN WORDS
flatenedwordsauth = [i for j in stemmedauth for i in j]
cleaned_auth = [" ".join(i) for i in stemmedauth]
#'==========================================PRE PROCESSING AND CLEANING COMPLETED FOR AUTHOR FEATURE========================
#'==========================================PRE-PROCESSING OF TEXT FEATURE========================
datalisttxt = list(features["text"])
datalisttxt = [i.lower() for i in datalisttxt] #LOWER
removepunchtxt = [re.sub("[^a-zA-Z\s]",'',str(word)) for word in datalisttxt] #REMOVE PUNCTUATIONS
removepunchtxt = [i.strip() for i in removepunchtxt] #REMOVE SPACES
datatokenisedtxt = [word_tokenize(i) for i in removepunchtxt] #TOKENIZE
datatokenisedtxt[1]
# STOP WORDS
removedstoptxt = [[i for i in j if i not in stop_words] for j in datatokenisedtxt]
# STEMMING
Porter = PorterStemmer()
stemmedtxt = [[Porter.stem(i) for i in j] for j in removedstoptxt]
#FLATEN WORDS
flatenedwordstxt = [i for j in stemmedtxt for i in j]
cleaned_txt = [" ".join(i) for i in stemmedtxt]

#'==========================================PRE PROCESSING AND CLEANING COMPLETED FOR TEXT FEATURE========================
features['title_updated'] = pd.DataFrame(cleaned_Title, index=features.index)
features['author_updated'] = pd.DataFrame(cleaned_auth, index=features.index)
features['text_updated'] = pd.DataFrame(cleaned_txt, index=features.index)
#=====================================================================================================================
# combining all three features using hstack
from scipy.sparse import hstack
features_final = features[['author_updated','title_updated','text_updated']]
[x_traintfidf, x_testtfidf, y_traintfidf, y_testtfidf] = train_test_split(features_final, target, random_state=None, stratify=None, train_size=0.7 )
vectortfidf1 = TfidfVectorizer(min_df=5, max_features=6000) # TF IDF for feature1
vectortfidf2 = TfidfVectorizer(min_df=5, max_features=6000) # TF IDF for feature2
vectortfidf3 = TfidfVectorizer(min_df=5, max_features=6000) # TF IDF for feature3
print(x_traintfidf['title_updated'].isnull().sum())
v1 = vectortfidf1.fit_transform(x_traintfidf['title_updated'])
v2 = vectortfidf2.fit_transform(x_traintfidf['author_updated'])
v3 = vectortfidf3.fit_transform(x_traintfidf['text_updated'])
v4 = vectortfidf1.transform(x_testtfidf['title_updated'])
v5 = vectortfidf2.transform(x_testtfidf['author_updated'])
v6 = vectortfidf3.transform(x_testtfidf['text_updated'])
Train_V123 = hstack([v1, v2, v3])
Test_V123 = hstack([v4, v5, v6])

Train_V123=Train_V123.toarray()
Test_V123=Test_V123.toarray()

# LOGISTIC REGRESSION MODEL TO PREDICT SENTIMENT - TF-IDF
modelogtfidf = LogisticRegression()
modelogtfidf.fit(Train_V123, y_traintfidf)
y_trainpredtfidf = modelogtfidf.predict(Train_V123)
y_testpredtfidf = modelogtfidf.predict(Test_V123)
print("Training Accuracy of tfidf vectoriser with logistic reg = ", accuracy_score(y_trainpredtfidf,y_traintfidf))
y_testpredtfidf = modelogtfidf.predict(Test_V123)
print("Test Accuracy of tfidf vectoriser with logistic reg = ", accuracy_score(y_testpredtfidf,y_testtfidf))

print(hasattr(vectortfidf1, 'idf_'))  # Should return True if fitted
print(hasattr(vectortfidf2, 'idf_'))  # Should return True if fitted
print(hasattr(vectortfidf3, 'idf_'))  # Should return True if fitted
import pickle

# Save each TfidfVectorizer
with open('tfidf_vectorizer1.pkl', 'wb') as file:
  pickle.dump(vectortfidf1, file)
with open('tfidf_vectorizer2.pkl', 'wb') as file:
  pickle.dump(vectortfidf2, file)
with open('tfidf_vectorizer3.pkl', 'wb') as file:
  pickle.dump(vectortfidf3, file)

# Save the Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as file:
  pickle.dump(modelogtfidf, file)
