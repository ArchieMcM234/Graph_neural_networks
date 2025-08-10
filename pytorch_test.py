import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics



BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)



class my_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1= nn.Conv2d(1, 32, 3)
        self.d1 = nn.Linear(26*26*32, 128)
        self.d2  = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.flatten(start_dim = 1)
        x = self.d1(x)
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out



learning_rate = 0.001
num_epochs = 5
device = torch.device("mps")
model = my_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss +=loss.detach().item()
        train_acc += (torch.argmax(logits, dim=1).flatten() == labels).type(torch.float).mean().item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_running_loss/len(trainloader):.4f}, Accuracy: {train_acc/len(trainloader):.4f}")
