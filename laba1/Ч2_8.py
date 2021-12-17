import numpy as np
import matplotlib.pyplot as plt


def arraytransformer(array, a, b):
    
    array=np.transpose(array)

    for i in range(0,b):
        mn=np.mean(array[i])
        cd = np.std(array[i])
        for j in range(0,a):
            array[i][j]=(array[i][j]-mn)/cd

    array=np.transpose(array)
    
    return array
    

a=7
b=2

array = np.zeros((a,b))

for i in range(0,a):
    array[i]=np.random.multivariate_normal(mean=[1,2],cov=[[2,1],[1,3]])

print(array)    
    
l = 1
for i in range(0,a):
    plt.plot(array[i][0], array[i][1],'ro')
    plt.annotate(l,xy=(array[i][0]+0.1, array[i][1]+0.1))
    l+=1

array=arraytransformer(array, a, b)

l = 1
for i in range(0,a):
    plt.plot(array[i][0],array[i][1],'bo')
    plt.annotate(l,xy=(array[i][0]+0.1, array[i][1]+0.1))
    l+=1

print(l) 
plt.show()