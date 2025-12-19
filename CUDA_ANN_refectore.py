import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import gc
import matplotlib.pyplot as plt

# Load your data
data_frames = torch.load(r"C:\Users\madhu\Desktop\data_frames.pt")
data_labels = torch.load(r"C:\Users\madhu\Desktop\data_labels.pt")

# Flatten the data frames for input into the ANN
flattened_data = data_frames.view(data_frames.size(0), -1)

# Create a dataset and split it into train and test sets
X_train, X_test, y_train, y_test = train_test_split(flattened_data, data_labels, test_size=0.2, random_state=42)

# Use a smaller batch size to reduce memory usage
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

# Define a smaller ANN model with mixed precision
class SimpleANN(nn.Module):
    def __init__(self, input_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x

# Use mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = flattened_data.size(1)
model = SimpleANN(input_size).to(device)

# Use automatic mixed precision (AMP) to save memory
scaler = torch.cuda.amp.GradScaler()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with gradient accumulation and mixed precision
def train_model(model, train_loader, criterion, optimizer, scaler, accumulation_steps=4, epochs=20):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device).float(), labels.to(device).float().view(-1, 1)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # No need to apply sigmoid before BCEWithLogitsLoss
                loss = criterion(outputs, labels) 

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
        
        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Empty the cache to clear up unused memory
        torch.cuda.empty_cache()
        gc.collect()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
    
    # Save the model after training
    torch.save(model.state_dict(), "model.pt")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Train the ANN model
train_model(model, train_loader, criterion, optimizer, scaler)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_accuracies = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.view(-1) == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Calculate test accuracy and visualize it
test_accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize the final test accuracy
plt.figure(figsize=(5, 5))
plt.bar(['Test Accuracy'], [test_accuracy], color='blue')
plt.ylim(0, 1)
plt.title('Test Accuracy')
plt.show()
