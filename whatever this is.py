import torch
import torch.nn as nn
import torch.optim as optim
import spikingjelly.activation_based as sj
from spikingjelly.activation_based import neuron
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
data_frames = torch.load(r"E:\Dissertation\data_frames.pt")  # Spike event data
data_labels = torch.load(r"E:\Dissertation\data_labels.pt")  # Corresponding labels

# Ensure the data is on the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_frames = data_frames.to(device)
data_labels = data_labels.to(device)

# Print the shapes of the data and labels for verification
print(f"Data frames shape: {data_frames.shape}")
print(f"Data labels shape: {data_labels.shape}")

# Define the Spiking Neural Network (SNN) model
class SpikingNet(nn.Module):
    def __init__(self):
        super(SpikingNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 6)  # 6 output neurons for 6 classes (letters 0 to 5)
        self.lif1 = sj.neuron.IFNode()  # Spiking neuron layer 1
        self.lif2 = sj.neuron.IFNode()  # Spiking neuron layer 2
        self.lif3 = sj.neuron.IFNode()  # Spiking neuron layer 3

    def forward(self, x):
        x = self.conv1(x)
        x = self.lif1(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = self.lif2(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.lif3(x)
        x = self.fc2(x)
        return x

# Combine the data into a TensorDataset
dataset = TensorDataset(data_frames, data_labels)

# Create a DataLoader for batching and shuffling the data
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and loss function
model = SpikingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the training function
def train(model, device, train_loader, optimizer, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Clear the gradients
            output = model(data)  # Forward pass
            loss = nn.CrossEntropyLoss()(output, target)  # Calculate the loss
            loss.backward()  # Backward pass (compute gradients)
            optimizer.step()  # Update model parameters
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Train the model for the specified number of epochs
train(model, device, train_loader, optimizer, epochs=10)

# Define the prediction function
def predict(model, device, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        data = data.to(device)
        output = model(data)  # Forward pass
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        return pred.item()  # Return the predicted class

# Example prediction
sample_data = data_frames[0].unsqueeze(0)  # Use a sample from the dataset for prediction
prediction = predict(model, device, sample_data)
print(f'Predicted class: {prediction}')
