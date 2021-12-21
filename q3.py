from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

phi = np.linspace(0, 20, 300)
r = 1 + phi

x = r * np.cos(phi)
y = r * np.sin(phi)
classes = np.zeros(300)
classes[:100] = 0
classes[100:200] = 1
classes[200:] = 2
points = np.column_stack((x + np.random.normal(scale=0.3,size = x.size), y + np.random.normal(scale=0.3,size=y.size)))

plt.plot(x,y)
plt.scatter(points[:,0], points[:,1], c=classes, cmap=matplotlib.colors.ListedColormap(['r','g','b']))
plt.gca().set_aspect('equal')

points_train, points_test, classes_train, classes_test = train_test_split(points,classes, test_size=0.2)

points_train, points_test = [torch.from_numpy(points_train).type(torch.FloatTensor), torch.from_numpy(points_test).type(torch.FloatTensor)]
classes_train, classes_test = [torch.from_numpy(classes_train).type(torch.long), torch.from_numpy(classes_test).type(torch.long)]

ds_train = TensorDataset(points_train, classes_train)
ds_val = TensorDataset(points_test, classes_test)

dl_train = DataLoader(ds_train, batch_size=10,)
dl_val = DataLoader(ds_val, batch_size=10)

device = torch.device('cpu')
class PointsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 3)
    def forward(self, x):
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return x

def check_accuracy(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval() 
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device) 
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

model = PointsNet()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
def train_model(model, criterion, epochs, optimizer, loader):
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            if i % 270 == 0:
                print('Epoch #', epoch, ', loss = ', loss.item())
                check_accuracy(dl_val, model)
                print()

train_model(model, torch.nn.CrossEntropyLoss(), 300, optimizer, dl_train)

x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))

model.eval()
Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.FloatTensor).to(device))
_, preds = Z.max(1)
preds = preds.cpu().numpy()

Z = preds.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(['r','g','b']))
plt.axis('off')
plt.show()