import wget
import os
import urllib.request


#wget.download('https://raw.githubusercontent.com/zaidalyafeai/zaidalyafeai.github.io/master/sketcher/mini_classes.txt')

f = open("mini_classes.txt","r")
# And for reading use
classes = f.readlines()
print(classes)
f.close()
classes = [c.replace('\n','').replace(' ','_') for c in classes]
print(classes)

#Download the Dataset
#os.remove("data/")       #先删除，再创建
#os.mkdir("data/")       #不能重复创建

def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        cls_url = c.replace('_', '%20')
        path = base + cls_url + '.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/' + c + '.npy')

download()