#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df= pd.read_csv(r"C:\Users\Administrator\OneDrive\Desktop\mushrooms.csv")
print(df)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[16]:


from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
for column in list(df.columns):
    df[column] = Le.fit_transform(df[column])


# In[18]:


df.dtypes


# In[21]:


X=df.drop('class',axis=1)
Y=df['class']


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=101)


# In[33]:


print(X_train.shape,X_test.shape)


# In[34]:


print(Y_train.shape,Y_test.shape)


# In[36]:


from sklearn.tree import DecisionTreeClassifier
dv=DecisionTreeClassifier()
dv.fit(X_train,Y_train)
Y_pred=dv.predict(X_test)


# In[37]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,Y_pred)


# In[38]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_pred)
accuracy


# In[ ]:




