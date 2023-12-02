import torch
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
import torch.nn as nn
import torch.optim as optim
from ChihuahuaMuffin import *
from chiMufData import *

train_path = 'archive(1)/train'
test_path = 'archive(1)/test'

train_set = ImageFolder(root=train_path,transform=transform)
test_set = ImageFolder(root=test_path,transform=transform)


train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)
test_test_loader = DataLoader(test_set,batch_size=1,shuffle=True)

# Example usage:
# Create an instance of the CNNBinaryClassifier
model = Classifier()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Example training loop (you need to replace this with your actual training data)
for epoch in range(num_epochs):
    for index, (inputs, labels) in enumerate(train_loader):  # Your dataloader for training data
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if index==10:
            break
        
    print('Epoch: {} Average Loss: {:.4f}'.format(epoch, loss / BATCH_SIZE))

print("-----------------------------")
print("Now Beginnning Testing.......")
print("-----------------------------")



model.eval()  # Set the model to evaluation mode
with torch.inference_mode():  # Disable gradient calculation during evaluation
    for batch_idx, (data,_) in enumerate(test_loader):
        outputs = model(data)
        loss = criterion(outputs, labels.float().view(-1, 1))
        if batch_idx==5:
            break

    average_loss = loss / BATCH_SIZE
    print(f"The test loss = {average_loss}")

correct = 0
incorrect = 0

with torch.inference_mode():
    for batch_idx, (data, label) in enumerate(test_test_loader):
        outputs = model(data)
        if round(float(outputs.view(-1))) == int(label.view(-1)):
            correct += 1
        else:
            incorrect +=1
print()
print("Correct = "+str(correct)+", Incorrect = "+str(incorrect))

correct_predictions = correct
incorrect_predictions = incorrect

# Calculate total predictions
total_predictions = correct_predictions + incorrect_predictions

# Calculate accuracy
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2%}")

torch.save(model.state_dict(), 'ChihuahuaMuffin.pth')

print('Model Saved.')
