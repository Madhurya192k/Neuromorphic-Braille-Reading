import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load your data
data_frames = torch.load(r"C:\Users\madhu\Desktop\data_frames.pt")
data_labels = torch.load(r"C:\Users\madhu\Desktop\data_labels.pt")

# Flatten the data frames for input into the ANN
flattened_data = data_frames.view(data_frames.size(0), -1)

# Create a dataset and split it into train and test sets
X_train, X_test, y_train, y_test = train_test_split(flattened_data, data_labels, test_size=0.2, random_state=42)

# Create DataLoader objects with a smaller batch size
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

# Define a smaller ANN model
class SimpleANN(nn.Module):
    def __init__(self, input_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Reduced layer size
        self.fc2 = nn.Linear(128, 64)          # Reduced layer size
        self.fc3 = nn.Linear(64, 32)           # Reduced layer size
        self.fc4 = nn.Linear(32, 16)           # Reduced layer size
        self.fc5 = nn.Linear(16, 1)            # Output layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# Use CPU instead of GPU to avoid memory issues
device = torch.device("cpu")

input_size = flattened_data.size(1)
model = SimpleANN(input_size).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with gradient accumulation
def train_model(model, train_loader, criterion, optimizer, accumulation_steps=8, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the ANN model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Calculate test accuracy
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')
