#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[4]:


dataset = pd.read_csv(r"C:\Users\Administrator\OneDrive\Desktop\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[5]:


print(X)


# In[6]:


print(y)


# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[8]:


print(X)


# In[14]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[15]:


print(X)


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[18]:


print(X_train)


# In[19]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


ann = tf.keras.models.Sequential()


# In[21]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[22]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[23]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[24]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[25]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 1)


# In[29]:


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# In[27]:


y_pred = ann.predict(X_test)
accuracy_score(y_test, y_pred)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[28]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




