import torch
import torch.nn as nn
import torch.optim as optim
import spikingjelly.activation_based as sj
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path

# Custom Dataset Class
class CharacterDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Placeholder: Update this to actually load your dataset
        data = []
        labels = []
        return data, labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        image = self.data[0][idx]
        label = self.data[1][idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# SNN Model
class SpikingNet(nn.Module):
    def __init__(self):
        super(SpikingNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 27)
        self.lif1 = sj.neuron_parametrized.IFNode()
        self.lif2 = sj.neuron_parametrized.IFNode()
        self.lif3 = sj.neuron_parametrized.IFNode()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lif1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.lif3(x)
        x = self.fc2(x)
        return x

# Training Setup
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing Setup
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')

def main():
    batch_size = 64
    learning_rate = 0.01
    epochs = 10

    train_data_path = './data/train'
    test_data_path = './data/test'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CharacterDataset(data_path=train_data_path, transform=transform)
    test_dataset = CharacterDataset(data_path=test_data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SpikingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()
