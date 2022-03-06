import sys
sys.path.append('../')  # 新加入的
import os
import numpy as np
import import2
from import2 import x_train,y_train,x_test,y_test,class_names
import matplotlib.pyplot as plt
from random import randint
from keras.models import load_model

#%matplotlib inline 不可用魔法函数，需要在后面加一个plt.show()
idx = randint(0,len(x_test))
img = x_test[idx]
plt.imshow(img.squeeze())
plt.show()
my_model = load_model('keras.h5')
pred = my_model.predict(np.expand_dims(img,axis=0))[0]
ind =(-pred).argsort()[:5]
latex = [class_names[x] for x in ind]
print(latex)

with open('class_names.txt','w') as file_handler :
    for item in class_names:
        file_handler.write("{}\n".format(item))


#https://stackoverflow.com/questions/53295570/userwarning-no-training-configuration-found-in-save-file-the-model-was-not-c
#涉及到load_model()的问题，好像是TensorFlow的optimizers不能被keras保存，解决方法就是需要重新compile一次
