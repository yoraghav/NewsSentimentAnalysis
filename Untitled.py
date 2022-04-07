import pandas as pd
import numpy as np

Doc = pd.read_csv("archive/all-data.csv",header=None)
lines = Doc[1]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(lines)
analyze = vectorizer.build_analyzer()
#print(len(vectorizer.get_feature_names()))
X.toarray()
final = pd.DataFrame(X.todense(),columns = vectorizer.get_feature_names())

final.sum()

final = final.drop(final.columns[final.sum() < 0.5], axis=1)
#final['Sentiment'] = Doc[0]

all_inputs = final[final.columns].values
all_labels = Doc[0].values


# from sklearn.model_selection import train_test_split
# 
# (training_inputs,
#  testing_inputs,
#  training_classes,
#  testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.2, random_state = True)

# from sklearn import tree
# classifier = tree.DecisionTreeClassifier()

# classifier.fit(training_inputs, training_classes)
# classifier.score(testing_inputs, testing_classes)

# In[13]:


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

encoder = LabelEncoder()
encoder.fit(all_labels)
encoded_Y = encoder.transform(all_labels)
dummy_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(64, input_dim=all_inputs.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(all_inputs,dummy_y, test_size=0.2,shuffle = True)

model.fit(X_train, y_train, epochs=30)

pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))   

