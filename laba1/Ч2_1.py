import numpy as np

def Calculatethefunction(a,c,b):
    
    return a @ c + b


list_vec1 = [2,2,3]
lict_vec2 = [2,1,2]
a = np.array(list_vec1)
c = np.array(lict_vec2)
b = 5
s = Calculatethefunction(a,c,b)
print(s)