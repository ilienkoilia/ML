import numpy as np
import networkx as nx

i = 0;

vk = nx.read_gml('VK.gml')
len_i = [0]*7

print("Количество уникальных пользователей равно: ", len(vk.nodes))

friends = {}
fv = []
for k in vk.nodes: 
    friends[k]=0

for e in vk.edges:
    friends[e[0]]+=1
    friends[e[1]]+=1
    
sorted_friends = sorted(friends.items(), key=lambda x: x[1], reverse=True)

print("Пользователи с наибольшим количеством друзей:")

for pair in sorted_friends[:15]:
    i+=1
    print("   ",i," Пользователь id ", pair[0], " c количеством друзей: ", pair[1])
    
for pair in sorted_friends:
    fv.append(pair[1])
    
print('Медианное число друзей:  ', np.median(fv))
print('Среднее число друзей: ', round(np.mean(fv)))

smallp = nx.all_pairs_smallp_length(vk)

for pair in smallp:
    for ln in pair[1].values():
        if ln>=1 and ln<=6:
            len_i[ln]+=1
        else:
            len_i[0]+=1

overall_ln=sum(len_i)            

for i in range(1,7):
    print(f'Доля пар с L={i} {len_i[i]/overall_ln}')          
print(f'Доля несвязанных пар или пар с L>6 или {len_i[0]/overall_ln}')