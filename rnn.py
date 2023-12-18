import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import torch.nn as nn
import numpy as np

# Load the Bitcoin dataset
df = pd.read_csv("./coin_Bitcoin.csv")

# Extract features (x) and target variable (y)
x = df[["High", "Low", "Open"]]
y = df["Close"]

# Standardize the features and target variable
scalar_x, scalar_y = StandardScaler(), StandardScaler()
x = scalar_x.fit_transform(x)
y = scalar_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

# Convert data to PyTorch tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
seq_len = 1
train_x = train_x.unsqueeze(1)

# Define a custom dataset class for PyTorch
class BitCoinDataSet(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        self.inputs = train_x
        self.target = train_y

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x, y = self.inputs[idx], self.target[idx]
        return x, y

# Hyperparameters
hidden_size = 128
num_layers = 2
learning_rate = 0.001
batch_size = 100
epoch_size = 10

# Create datasets and data loaders
train_dataset = BitCoinDataSet(train_x, train_y)
test_dataset = BitCoinDataSet(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the Recurrent Neural Network (RNN) model
class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_feature_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out.squeeze())
        out = out.squeeze()
        out = self.relu(out)
        return out

# Initialize RNN model, loss function, and optimizer
torch.manual_seed(0)
rnn = RNN(input_feature_size=3, hidden_size=hidden_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# Training loop
rnn.train()
for epoch in range(epoch_size):
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = rnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluation on the test set
prediction = []
ground_truth = []

rnn.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data

        ground_truth += targets.flatten().tolist()
        out = rnn(inputs).detach().flatten().tolist()
        prediction += out

# Inverse transform the predictions and ground truth to original scale
prediction = scalar_y.inverse_transform(np.array(prediction).reshape(-1, 1)).flatten()
ground_truth = scalar_y.inverse_transform(np.array(ground_truth).reshape(-1, 1)).flatten()

# Calculate R-squared score
r2score = r2_score(prediction, ground_truth)

# Print R-squared score
print(r2score)
