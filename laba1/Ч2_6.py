import numpy as np
import random

def Calculate_the_angle(a,b):
    a_c = (b / np.linalg.norm(a))
    b_c = (a / np.linalg.norm(b))
    rangle = np.arccos(np.clip(np.dot(a_c, b_c), -2.0, 2.0))
    nook = np.degrees([rangle.real])[0]
    return nook


    
vec = []
A = 1000
d = 6
for i in range(A):
    list = []
    for j in range(d):
        list.append( random.randint(-20, 20) )
    vec.append( np.array(list) )

list = []
for j in range(d):
        list.append( random.randint(-10, 10) )
b =  np.array(list) 

nook_90 = nook_30 = 0
for i in range(A):
    nook = Calculate_the_angle(vec[i],b)
    if nook < 90:
        nook_90 += 1
    if nook < 30:
        nook_30 += 1
        
print(nook_90/A)
print(nook_30/A)