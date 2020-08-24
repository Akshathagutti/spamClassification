import pandas as pd
import numpy as np
import re
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
df_spam = pd.read_csv("spam.csv", usecols = ["label",'messages'])
ps = PorterStemmer()
corpus = []
for i in range(0, len(df_spam)):
    review = re.sub('[^a-zA-z]',' ',df_spam['messages'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
spam_cv = CountVectorizer(max_features=5000)
X = spam_cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df_spam['label'])
y = y.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)
print(y_pred)

from  sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_pred)
print(accuracy)

