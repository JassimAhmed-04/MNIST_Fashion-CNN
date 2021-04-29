#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, cross_validate


# # Importing the Dataset

# In[54]:


mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# # Reshaping, Feature Scaling and Transforming the Data

# In[55]:


#reshape data from 3-D to 2-D array
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
#feature scaling
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
#fit and transform training dataset
X_train = minmax.fit_transform(X_train)
#transform testing dataset
X_test = minmax.transform(X_test)
print('Number of unique classes: ', len(np.unique(y_train)))
print('Classes: ', np.unique(y_train))


# # Visualizing the Data

# In[56]:


fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(15,5))          
ax = axes.ravel()
for i in range(10):
    ax[i].imshow(X_train[i].reshape(28,28))
    ax[i].title.set_text('Class: ' + str(y_train[i]))              
plt.subplots_adjust(hspace=0.5)                                    
plt.show()


# # Model Building

# In[57]:


#initializing CNN model
classifier_e26 = Sequential()
#add 1st hidden layer
classifier_e26.add(Dense(input_dim = X_train.shape[1], units = 256, kernel_initializer='uniform', activation='relu'))
#add output layer
classifier_e26.add(Dense(units = 10, kernel_initializer='uniform', activation='softmax'))
#compile the neural network
classifier_e26.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model summary
classifier_e26.summary()


# In[58]:


#fit training dataset into the model
classifier_e26_fit = classifier_e26.fit(X_train, y_train, epochs=26, verbose=0)


# # Evaluating the Model

# In[59]:


#evaluate the model for testing dataset
test_loss_e26 = classifier_e26.evaluate(X_test, y_test, verbose=0)
#calculate evaluation parameters
f1_e26 = f1_score(y_test, class`ifier_e26.predict_classes(X_test), average='micro')
roc_e26 = roc_auc_score(y_test, classifier_e26.predict_proba(X_test), multi_class='ovo')
#create evaluation dataframe
stats_e26 = pd.DataFrame({'Test accuracy' :  round(test_loss_e26[1]*100,3),
                      'F1 score'      : round(f1_e26,3),
                      'ROC AUC score' : round(roc_e26,3),
                      'Total Loss'    : round(test_loss_e26[0],3)}, index=[0])
#print evaluation dataframe
display(stats_e26)

