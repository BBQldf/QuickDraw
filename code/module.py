import keras
import tensorflow as tf
from keras import layers
import sys
sys.path.append('../')  # 新加入的
import import2
from import2 import x_train,y_train,x_test,y_test,class_names

model =keras.Sequential()
model.add(layers.Convolution2D(16,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Convolution2D(32,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Convolution2D(64,(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(100,activation='softmax'))

#模型训练
adam = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['top_k_categorical_accuracy'])
print(model.summary())

#训练命令下达
model.fit(x=x_train,y=y_train,validation_split=0.1,batch_size=256,verbose=2,epochs=5)


score=model.evaluate(x_test,y_test,verbose=0)
print('Test accuarcy:{:0.2f}%'.format(score[1]*100))

model.save('keras.h5')


#os.mkdir('model/')