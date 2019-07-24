#!/usr/bin/env python
# coding: utf-8

# In[50]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:





# [ 데이터에 대한 설명 ]
# 
# MNIST 데이터셋은 미국 국립표준기술원(NIST)이 고등학생과 인구조사국 직원 등이 쓴 손글씨를
# 이용해 만든 데이터로 구성되어 있므며, 70,000개의 글자 이미지에 각각 0부터 9까지 레이블을
# 붙인 데이터셋으로 케라스 내에서 제공하는 데이터셋입니다.
# 케라스의 MNIST 데이터는 10개의 클래스, 총 70,000개의 이미지 중 60,000개를 학습용으로,
# 10,000개를 테스트용으로 미리 구분해 놓고 있습니다. 이미지는 가로 28 × 세로 28 = 총 784개
# 의 픽셀로 이루어져 있고, 이미지의 각 픽셀은 밝기 정도에 따라 0부터 255까지의 그레이 스케일
# 정보를 가지고 있습니다
# 
# 
# 
# [ 구현에 대한 설명 ].
# 학습된 모델을 파일("/content/drive/My Drive/colab_myworks/models/{epoch:02d}-
# {val_loss:.4f}.hdf5")로 저장하고 모델의 최적화 단계에서 학습을 자동 중단하게끔 설정하고, 10회
# 이상 모델의 성능 향상이 없으면 자동으로 학습을 중단하도록 설계합니다.
# 배치 크기를 200, 학습주기를 30으로 설정하여, 테스트셋으로 최종 모델의 성과를 측정하여 그
# 값을 출력합니다.
# 학습주기와 손실에 대한 결과를 그래프로 표현합니다.
# 아래의 설계와 같이 컨볼루션 신경망을 얹어 딥러닝
# 
# ![image.png](attachment:image.png)
# 
# 

# In[ ]:


from keras.datasets import mnist


# In[ ]:


(X_train, y_train), (X_test, y_test) = mnist.load_data ()  #array정보로 데이터 읽어오기


# In[53]:


print (X_train.shape)
print (X_test.shape)


# In[54]:


print (y_train.shape)
print (y_test.shape)


# In[ ]:


# tensor 변환 및 전환 tensor에 맞게 데이터 변환
import numpy as np
X_train = X_train.reshape (X_train.shape[0], 28, 28, 1).astype (np.float32) / 255   #0~255 인데 이것을 255로 나누었기때문에 범주가 0~1로
X_test = X_test.reshape (X_test.shape[0], 28, 28, 1).astype (np.float32) / 255


# In[ ]:


# Class label One hot encoding (이미 tensor 변환된 데이터를 이용하여 수치형 데이터를 범주형데이터로 바꿈)


from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[57]:


y_train.shape    #(samples, labels)


# In[58]:


y_train[0]


# In[ ]:


# 모델 구성
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


# In[ ]:


model = models.Sequential()

model.add(layers.Conv2D (32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

model.add(layers.Conv2D (64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D (pool_size = 2))
          
model.add(layers.Dropout(0.25))
          
model.add(layers.Flatten())
          
model.add(layers.Dense (128,activation = 'relu'))
          
model.add(layers.Dropout (0.5))
          
model.add(layers.Dense (10,activation = 'softmax'))
          
          
model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['acc'])


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


# early stopping 관련 call back 만들기 
#ModelCheckpoint 콜백 함수는 Keras에서 모델을 학습할 때마다 중간중간에 콜백 형태로 알려줍니다


model_path ="/content/drive/My Drive/Colab_myworks/models/{epoch:02d}-{val_loss:.4f}.hdf5"
check_pointer = ModelCheckpoint(filepath = model_path, monitor = 'val_loss'
                               , verbose = 1, save_best_only = True)
early_stopper = EarlyStopping (monitor = 'val_loss', patience = 10)


# In[63]:


history = model.fit (X_train, y_train, validation_data = (X_test, y_test),
                    epochs = 30, batch_size = 200, verbose = 0, callbacks = [early_stopper, check_pointer])


# In[64]:


model.evaluate (X_test, y_test)[1]


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


y_val_loss = history.history['val_loss']


# In[ ]:


y_loss = history.history['loss']


# In[ ]:


x_epoch = np.arange (len(y_loss))


# In[69]:


plt.plot(x_epoch, y_val_loss, marker = '.', c = 'red', label = 'validation loss')
plt.plot(x_epoch, y_loss, marker = '.', c = 'blue', label = 'train loss')
plt.legend()


# In[ ]:




