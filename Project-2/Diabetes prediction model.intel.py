#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[18]:


dataset=pd.read_csv(r"C:\Users\Administrator\OneDrive\Desktop\Diabetes Prediction.csv")
x=dataset.iloc[:,0:-1]
y=dataset.iloc[:,-1].values


# In[19]:


print(dataset)


# In[20]:


print(x)


# In[21]:


print(y)


# In[22]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[24]:


print(y_test)


# In[10]:


ann = tf.keras.models.Sequential()


# In[11]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[12]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# In[13]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[14]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[25]:



ann.fit(x_train, y_train,batch_size=40,epochs=30)


# In[27]:


y_pred = ann.predict(x_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[150]:


print(y_pred)


# In[28]:


print(y_test)


# In[33]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[ ]:




