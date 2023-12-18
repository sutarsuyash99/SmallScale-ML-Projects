import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder('./petimages', transform=transform)

# Split the dataset into train and test sets
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

# Set batch size and create data loaders
batch_size = 32
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(16928, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the CNN model
cnn = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)

# Set the model to training mode
cnn.train() 
epoch_size = 4

# Training loop
for epoch in range(epoch_size):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluation on the test set
ground_truth = []
predictions = []

# Set the model to evaluation mode
cnn.eval() 
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs, 1)
        ground_truth += labels.tolist()
        predictions += predicted.tolist()

# Calculate and print metrics
accuracy = accuracy_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)

print(f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')
