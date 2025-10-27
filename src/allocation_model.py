"""
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints:    python train.py data_dir --save_dir save_directory
Choose architecture:                  python train.py data_dir --arch "vgg13"
Set hyperparameters:                  python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training:                 python train.py data_dir --gpu
"""

# TODO: IMPORTS HERE
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

#######################################################################################################################
# TODO: Add command line arguments

# Set default values for all future arguments
default_hidden_units = [1024, 512]
default_learning_rate = 0.003
default_epochs = 5
default_arch = 'densenet161'
default_device = 'gpu'

# Creates Argument Parser object called parser
parser = argparse.ArgumentParser()

# TODO: Arguments for inputs
parser.add_argument('data_dir', type=str, help='Location of directory with data for training')
parser.add_argument('--save_dir', type=str, help='Set directory to save checkpoint')
parser.add_argument('--arch', type=str, default=default_arch,
                    help='Select a pretrained networks: "densenet161" or "alexnet"')
parser.add_argument('--hidden_units', type=int, default=default_hidden_units, help='List of hidden units'
                                                                                   'of model (max 2)')
parser.add_argument('--learning_rate', type=float, default=default_learning_rate,
                    help='Set learning rate hyperparameter (float)')
parser.add_argument('--epochs', type=int, default=default_epochs, help='Set epochs hyperparameter (int)')
parser.add_argument('--gpu', type=str, default=default_device, help='Uses "GPU" if available')

# Assign variable in parse_args() to access the arguments in the argparse object
args = parser.parse_args()

# Assign variables to use in the model
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
hidden_units = args.hidden_units
learning_rate = args.learning_rate
epochs_train = args.epochs
device = args.gou

#######################################################################################################################
# TODO: DEFINE THE MODEL

# Use getattr() f'n to get the arch model from Class 'models'
model = getattr(models, arch)(pretrained=True)

## Freeze 'feature' parameters
for param in model.parameters():
    param.requires_grad = False

# TODO: Define the input size, with the output fixed at 102
if arch == 'densenet161':
    input_size = 2208
elif arch == 'alexnet':
    input_size = 9216
else:
    print('Error, unexpected architecture set')
    exit()

# TODO: Define Model Architecture
model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.15),
                                 nn.Linear(hidden_units[0], hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units[1], 102),
                                 nn.LogSoftmax(dim=1))

# Define criterion for the loss
criterion = nn.NLLLoss()

# Define optimizer for backpropagation
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Move weights to memory of the active device (GPU/CPU)
model.to(device)

print("Model successfully defined")

#######################################################################################################################
# TODO:  DEFINE TRAINING AND VALIDATION DATALOADERS
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(43),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(p=0.3),
                                       transforms.RandomVerticalFlip(p=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=36, shuffle=True)
validatorloader = torch.utils.data.DataLoader(valid_dataset, batch_size=36)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=36)

print("3 sets of datasets successfully created: Training, Validation (during training), Testing")

#######################################################################################################################
# TODO: TRAIN NETWORK and VALIDATE

# Set epochs hyperparameter & Initial running loss
epochs = epochs_train
running_loss = 0

# Initialize # of steps for Eval check
steps = 0
print_every = 50

for e in range(epochs):
    print(f"For loop 1: {e} and {epochs}")

    # Training for pass for each epoch
    for images, labels in trainloader:

        # Keep track of 'steps' for testing 'if' clause
        steps += 1

        # Move images, labels tensors to 'device' memory
        images, labels = images.to(device), labels.to(device)

        # Clean accumulated gradients from last pass
        optimizer.zero_grad()

        # Forward pass, loss calc, backward pass, update weights
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Check periodically to run validation pass, in the middle
        if steps % print_every == 0:

            # Initialize test loss and accuracy variables
            valid_loss = 0
            accuracy = 0

            # Turn off gradients for Eval pass of the model
            model.eval()

            # Turn off gradients for Eval pass of the model
            with torch.no_grad():

                # Start Eval pass w/ its own set of images w/in the batch
                for valid_img, valid_labels in validatorloader:
                    # Move to valid_img, valid_labels tensors to 'device' memory
                    valid_img, valid_labels = valid_img.to(device), valid_labels.to(device)

                    # 1 forward pass, loss calc for the batch of images
                    log_ps = model(valid_img)
                    batch_loss = criterion(log_ps, valid_labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    # Convert log(softmax) prob to prob distribution
                    ps = torch.exp(log_ps)

                    # Get 'top_class' from 'topk' and equivocate the prediction
                    # and valid_labels tensor (i.e. same dimensions)
                    # 1 = 1st largest value in prob distribution
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == valid_labels.view(*top_class.shape)

                    # Convert 'equals' to a FloatTensor, then calc mean
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print data on this Training and Validation pass
            print(f"Epoch {e + 1}/{epochs}.. "
                  f"Training loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {valid_loss / len(validatorloader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(validatorloader):.3f}")

            # Reset running loss to 0
            running_loss = 0

            # Set model back to training mode
            model.train()

print("Model trained, please see above for statistics on the training")

#######################################################################################################################
# TODO: SAVE CHECKPOINT

model.class_to_idx = train_dataset.class_to_idx

checkpoint = {'input_size': input_size,
              'output_size': 102,
              'hidden_units': hidden_units,
              'network': arch,
              'classifier': model.classifier,
              'learning_rate': learning_rate,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'epochs': epochs_train,
              'class_to_idx': model.class_to_idx
              }

torch.save(checkpoint, 'checkpoint.pth')

print("Checkpoint successfully saved")
