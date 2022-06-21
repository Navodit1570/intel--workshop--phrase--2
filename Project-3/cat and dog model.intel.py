#!/usr/bin/env python
# coding: utf-8

# In[270]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[271]:


tf.__version__


# In[272]:


train_dta=ImageDataGenerator(rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip= True)
training_dataset =train_dta.flow_from_directory('C:/Users/Administrator/OneDrive/Desktop/New folder',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
                                                


# In[273]:


test_dta=ImageDataGenerator(rescale=1./255)
test_dataset=test_dta.flow_from_directory('C:/Users/Administrator/OneDrive/Desktop/New folder/dataset cat and dog',
                                          target_size=(64,64),
                                          batch_size=32,
                                          class_mode='binary')


# In[274]:


cnn = tf.keras.models.Sequential()


# In[275]:


cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))


# In[276]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


# In[277]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[278]:


cnn.add(tf.keras.layers.Flatten())


# In[279]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[280]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[281]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[282]:


cnn.fit(x = training_dataset, validation_data = test_dataset, epochs = 2)


# In[283]:


import numpy as np
from keras.preprocessing import image
test_image = tf.keras.utils.load_img('C:/Users/Administrator/OneDrive/Desktop/dog.jpg', target_size = (64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_dataset.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[284]:


print(prediction)


# In[285]:


cnn.summary()


# In[ ]:




