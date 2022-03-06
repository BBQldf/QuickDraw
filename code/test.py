import numpy as np

for i in [0, 1]:
    print(i);
x=np.zeros((3,3),dtype=np.float)
print(x)

stroke_lengths = [4,5,6,7,8,9,10]
total_points = sum(stroke_lengths)
print(total_points)

L = ['haha','xixi','hehe','heihei','gaga']
print(L[1:]);
print(L[0:-1])