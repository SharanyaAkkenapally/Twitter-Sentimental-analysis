
import numpy as np
import pandas as pd

dataset=pd.read_csv("twitter dataset.csv")
dataset.isnull().any()
dataset = dataset.dropna()
dataset=dataset.head(7000)
dataset.reset_index(drop=True,inplace=True)
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
 
data=[]
#as dataset size is(27481,2)
for i in range(0,7000):
    review=dataset["selected_text"][i]

    review=re.sub('[^a-zA-z]'," ",review)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]

    review=' '.join(review)
    
    data.append(review)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["sentiment"]=le.fit_transform(dataset["sentiment"])



from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000) #given some random number file has 1565 unique words
x=cv.fit_transform(data).toarray()
y=dataset.iloc[:,3:].values
from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder()
a=one.fit_transform(y[:,0:]).toarray()
y=a
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, random_state = 0,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=5917,activation="relu",init="uniform"))
model.add(Dense(units=3000,activation="relu",init="uniform"))
model.add(Dense(units=3000,activation="relu",init="uniform"))
model.add(Dense(units=3000,activation="relu",init="uniform"))
model.add(Dense(units=3,activation="softmax",init="uniform"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=1,batch_size=32)

y_pred=model.predict(x_test)
l=["negative","neutral","positive"]

y_p2=model.predict_classes(cv.transform(["nooooo"]))
print(l[y_p2[0]])

y_p1=model.predict_classes(cv.transform(["want it back"]))
print(l[y_p1[0]])
 
y_p1=model.predict_classes(cv.transform(["beauty"]))
print(l[y_p1[0]])
