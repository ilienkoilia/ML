import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
n_epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


train_val_data = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test_data = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])) 

train_data, val_data = train_test_split(train_val_data, test_size=10000, random_state=42)

torch.backends.cudnn.enabled = False

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=64,
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=64,
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=64,
                                          shuffle=False)
examples = enumerate(test_loader)




batch_idx, (example_data, example_targets) = next(examples)
example_data.shape




fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net()

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []

val_losses = []


test_losses = []



def val():   
    network.eval()    
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = network(data)
            val_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print('\nVal set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    val_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))
    
def test(): 
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():# Спецпосланнику не нужно возвращать градиент
        for data, target in test_loader:
            output = network(data)
            # Рассчитать убыток
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # Рассчитать точное число
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')
            val()

for epoch in range(1, n_epochs + 1):
  train(epoch)

print("Training completed")
test()    


def loss_plot( loss,  vloss):
    plt.plot(loss)
    plt.plot(vloss)
    
loss_plot( train_losses, val_losses)



def predict_image(image, model):
    xb = image.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()



image, label = val_data[111]
print('Label:', label, ', Predicted:', predict_image(image, network))
predict_image(image, network)
plt.imshow(image[0], cmap='gray')



n = 10
m = 10

imcount = 0


cls_err = [[0] * m for i in range(n)]

cls_data = [[],[]]

for i in range(len(val_data)):
    img, label = val_data[i]
    prediction = predict_image(img, network)
    cls_data[0].append(label)
    cls_data[1].append(prediction)
    if label != prediction:
        cls_err[label][prediction]+=1
        if imcount<6:
            imcount+=1;
            plt.imshow(img[0], cmap='gray')
            plt.show()
            print('Label:', label, ', Predicted:', predict_image(img, network))

print(classification_report(cls_data[0], cls_data[1]))


cls_err_np = np.array(cls_err)

def mx_pairs(x):
    x = x + np.transpose(x)
    x = np.triu(x)
    for i in range(5):
        max = x[0][0]
        for i in range(len(x)):
            for j in range(len(x[i] )):
                if x[i][j] > max:
                    max =  x[i][j]

        list_index_max =[ (i,j) for i in range(len(x))  for j in range(len(x[i])) if x[i][j]  == max]
        line, column = list_index_max[0]
        elem = [line,column,max]
        print('Чаще всего путают: ')
        print(f'Цифры {elem[0]} и {elem[1]} {elem[2]} количество раз')
        x[elem[0]][elem[1]]=0


mx_pairs(cls_err_np)



