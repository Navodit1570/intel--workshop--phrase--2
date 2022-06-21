#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[2]:


tf.__version__


# In[3]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/Administrator/OneDrive/Desktop/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[5]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/Administrator/OneDrive/Desktop/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary'),
                                            


# In[6]:


cnn = tf.keras.models.Sequential()


# In[7]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[8]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[9]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[10]:


cnn.add(tf.keras.layers.Flatten())


# In[11]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[12]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[13]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[14]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 2)


# In[18]:


import numpy as np
from keras.preprocessing import image
test_image = tf.keras.utils.load_img('C:/Users/Administrator/OneDrive/Documents/mask image.jpg', target_size = (64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'with out mask'
else:
  prediction = ' mask'


# In[19]:


print(prediction)


# In[20]:


cnn.summary()


# In[ ]:




