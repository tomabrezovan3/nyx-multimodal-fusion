import os # lets us work with folders and file paths
import torch # main PyTorch library
import torch.nn as nn # neural network layers and loss functions
import torch.optim as optim # optimizers like Adam
from torchvision import datasets, transforms, models # image datasets, transforms, pretrained models
from torch.utils.data import DataLoader # creates batches and feeds data to the model

# -----------
# 1. Settings
# -----------
DATA_DIR = 'data_synth_ir' # main dataset folder
BATCH_SIZE = 16 # number of images the model sees at once in one batch
IMAGE_SIZE = 128 # every image will be resized to 128x128
EPOCHS = 5 # how many full passes through the training dataset
LEARNING_RATE = 1e-3 # how big each weight update step is
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# If a CUDA GPU is available, use it
# Otherwise use CPU

# ------------
# 2. Transform
# ------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # Your IR images are probably single-channel grayscale
    # ResNet expects 3 channels (like RGB), so this copies grayscale information into 3 channels

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # Resize every image to find a fixed size: 128x128
    # Neural networks need consistent input size

    transforms.ToTensor(),
    # Converts image from PIL format / numpy style into a PyTorch tensor
    # Also scales pixel values from [0,255] to roughly [0,1]
])

# -----------------------
# 3. Datasets and loaders
# -----------------------

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
# Reads images from 'data/train/'
# Each subfolder inside 'train/' becomes a class label
# Example:
# 'data/train/0/...'
# 'data/train/1/...'

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
# Validation set: used during training to check how well the model generalizes

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
# Test set: used only at the very end for final evaluation

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# DataLoader breaks the dataset into batches
# shuffle=True means training images are mixed each epoch, which helps training

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# No need to shuffle validation data because we are only evaluating

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Same idea for test data

print('Classes:', train_dataset.classes)
# Shows the class folder names, e.g. ['0', '1']

print('Train samples:', len(train_dataset))
print('Val samples:', len(val_dataset))
print('Test samples:', len(test_dataset))
# Shows how many images are in each split

print('Using device:', DEVICE)
# Tells you whether training runs on CPU or GPU

# --------
# 4. Model
# --------

model = models.resnet18(weights=None)
# Creates a ResNet18 convolutional neural network
# weights=None means:
# start from random weights instead of pretrained ImageNet weights

model.fc = nn.Linear(model.fc.in_features, 2)
# Replace the final classification layer
# Original ResNet18 outputs 1000 classes
# We change it to output 2 classes for your binary classifier

model = model.to(DEVICE)
# Move model to CPU or GPU

# ---------------------
# 5. Loss and optimizer
# ---------------------

criterion = nn.CrossEntropyLoss()
# CrossEntropyLoss is standard for classifiers
# It compares the model's raw outputs ('logits') to the true class label

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Adam is the algorithm that updates the model weights
# model.parameters() means: optimize all trainable weights in the model
# lr=LEARNING_RATE controls how big the updates are

# ------------------
# 6. Train one epoch
# ------------------

def train_one_epoch():
    model.train()
    # Put model in training mode
    # Important because some layers behave  differently in training vs evaluation (like dropout or batch norm)

    running_loss = 0.0
    # total loss accumulated over the whole epoch

    correct = 0
    # total number of correct predictions

    total = 0
    # total number of samples seen

    for images, labels in train_loader:
        # Loop over one batch at a time
        # images = batch of input images
        # labels = true class IDs for those images

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # Move batch data to same device as model

        optimizer.zero_grad()
        # Clear old gradients from the previous step
        # If you do not do this, gradients keep accumulating

        outputs = model(images)
        # Forward pass:
        # send images through the network to get predictions
        # outputs shape is usually [batch_size, num_classes]

        loss = criterion(outputs, labels)
        # Compare predictions with true labels to compute how wrong the model is

        loss.backward()
        # Backward pass:
        # compute gradients of the loss with respect to all trainable weights

        optimizer.step()
        # Update model weights using those gradients

        running_loss += loss.item() * images.size(0)
        # loss.item() gives the scalar loss value for this batch
        # Multiply by batch size so we can later compute correct average epoch loss

        preds = outputs.argmax(dim=1)
        # Choose the class with the highest score for each image
        # Example: if outputs = [2.1, 0.3], predicted class is 0

        correct += (preds == labels).sum().item()
        # Count how many predictions in this batch were correct

        total += labels.size(0)
        # Add number of samples in this batch to total

    epoch_loss = running_loss / total
    # average  loss over the whole epoch

    epoch_acc = correct / total
    # accuracy over the whole epoch

    return epoch_loss, epoch_acc

# -----------
# 7. Evaluate
# -----------

def evaluate(loader):
    model.eval()
    # Put model in evaluation mode
    # This disables training-specif behavior

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # Disables gradient calculation
        # Saves memory and computation because we are not training here

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Move validation/test batch to device

            outputs = model(images)
            # Forward pass only

            loss = criterion(outputs, labels)
            # Compute loss for monitoring

            running_loss += loss.item() * images.size(0)
            # Accumulate total loss

            preds = outputs.argmax(dim=1)
            # Predicted class for each image

            correct += (preds == labels).sum().item()
            # Count correct predictions

            total += labels.size(0)
            # Count samples

    epoch_loss = running_loss / total
    # average loss over dataset

    epoch_acc = correct / total
    # average accuracy over dataset

    return epoch_loss, epoch_acc

# ----------------
# 8. Training loop
# ----------------

for epoch in range(EPOCHS):
    # Repeat training + validation for the chosen number of epochs

    train_loss, train_acc = train_one_epoch()
    # Train on the training set once

    val_loss, val_acc = evaluate(val_loader)
    # Evaluate on validation set after that epoch

    print(
        f'Epoch {epoch+1}/{EPOCHS} | '
        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
    )
    # Print progress so you can see if model is improving or overfitting

# -------------
# 9. Final test
# -------------

test_loss, test_acc = evaluate(test_loader)
# After all training is finished, evaluate once on the test set

print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
# Print final performance on unseen test data

# --------------
# 10. Save model
# --------------

torch.save(model.state_dict(), 'ir_classifier.pth')
# Save learned weights only
# Later, you can recreate the same model structure and load these weights back in

print('Model saved to ir_classifier.pth')


