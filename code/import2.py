import os
import glob
import numpy as np
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow import keras
import tensorflow as tf
from keras import layers

def load_data(root,vfold_ratio=0.2,max_items_per_class=4000):
    all_files = glob.glob(os.path.join(root,'*.npy'))

    #初始化变量 x y
    x=np.empty([0,784])
    y=np.empty([0])
    class_names = []

    #读入data文件
    for idx,file in enumerate(all_files):
        data = np.load(file)
        data = data[0:max_items_per_class, : ]
        labels = np.full(data.shape[0],idx)

        x=np.concatenate((x,data),axis=0)
        y=np.append(y,labels)

        class_name,ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

    data = None
    labels=None

    #数据随机化
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, : ]
    y = y[permutation]

    #数据分为训练集和测试集
    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))

    x_test=x[0:vfold_size, :]
    y_test=y[0:vfold_size]

    x_train=x[vfold_size:x.shape[0], :]
    y_train =y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names


x_train,y_train,x_test,y_test,class_names = load_data('data/')
num_classes = len(class_names)
image_size = 28
print(len(x_train))

#显示随机数据
from matplotlib import pyplot as plt
from  random import randint
idx =randint(0,len(x_train))
plt.imshow(x_train[idx].reshape(28,28))
plt.show()
print(class_names[int(y_train[idx].item())])

#对数据进行处理
x_train=x_train.reshape(x_train.shape[0],image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0],image_size,image_size,1).astype('float32')

x_train /= 255.0
x_test /= 255.0

#将类向量转换为类矩阵
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)


# #模型构建
# model =keras.Sequential()
# model.add(layers.Convolution2D(16,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Convolution2D(32,(3,3),padding='same',activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Convolution2D(64,(3,3),padding='same',activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128,activation='relu'))
# model.add(layers.Dense(100,activation='softmax'))
#
# #模型训练
# adam = tf.train.AdamOptimizer()
# model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['top_k_categorical_accuracy'])
# print(model.summary())
#
# #训练命令下达
# model.fit(x=x_train,y=y_train,validation_split=0.1,batch_size=256,verbose=2,epochs=5)
#
#
# score=model.evaluate(x_test,y_test,verbose=0)
# print('Test accuarcy:{0.2f}%'.format(score[1]*100))

