
# coding: utf-8

# # importing tensorflow 

# In[2]:


import tensorflow as tf


# # Importing the Keras libraries and packages

# In[3]:


from tensorflow.contrib.keras import layers


# In[4]:


from tensorflow.contrib.keras import models


# In[5]:


classifier=models.Sequential()


# # convolution

# In[6]:


classifier.add(layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# # pooling

# In[7]:


classifier.add(layers.MaxPool2D(pool_size=(2,2)))


# # second convolutional layer

# In[8]:


classifier.add(layers.Conv2D(32,(3,3),activation='relu'))
classifier.add(layers.MaxPool2D(pool_size=(2,2)))


# # flattening

# In[9]:


classifier.add(layers.Flatten())


# # fully connected layers

# In[10]:


classifier.add(layers.Dense(units=128,activation='relu'))


# In[11]:



classifier.add(layers.Dense(units=1,activation='sigmoid'))


# # compiling the model

# In[12]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# # fitting cnn to the model

# In[13]:


from tensorflow.contrib.keras import preprocessing


# In[14]:


train_datagen = preprocessing.image.ImageDataGenerator(
                                                    rescale=1./255,
                                                    shear_range=0.2,
                                                    zoom_range=0.2,
                                                    horizontal_flip=True)


# In[15]:


test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[16]:


training_set = train_datagen.flow_from_directory( 'convo/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')


# In[16]:


test_set = test_datagen.flow_from_directory('convo/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')


# In[ ]:


classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
                        

