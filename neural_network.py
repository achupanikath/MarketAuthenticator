import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from shutil import copyfile
#import pandas as pd
#from torchsummary import summary
#from torchvision import datasets, transforms

BATCH_SIZE = 64
INPUT_SIZE = 224
HIDDEN_LAYER_SIZE = INPUT_SIZE*2
NUM_CLASSES = 2
EPOCHS = 15

# most of this code is adapted from the PyTorch documentation on resnet
# found here: https://pytorch.org/hub/pytorch_vision_resnet/


class Net(nn.Module):
    # adapted from https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.layer2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.layer3 = nn.Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES)

    # x represents our data
    def forward(self, x):
        # Use the rectified-linear activation function over x
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        # Apply softmax to x
        output = F.softmax(x, dim=0)
        return output


def make_ml_model():
    model = Net()
    return model


def save_model(model, path="pytorch_resnet_saved"):
    #save a model
    path = os.path.join(os.getcwd(), path)
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model(filename):
    # load a model from a file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(filename, map_location=device)
    model.eval()
    return model


def make_val_set(p = 0.2):
    # takes a random subset of the kaggle data and creates a validation set
    # the validation images are removed from the training data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.getcwd()
    classes = os.listdir(os.path.join(PATH, "Train"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Validation", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Train", sign))
        random.shuffle(list_of_files)
        n_idxs = int(len(list_of_files)*p)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            os.rename(os.path.join(PATH, "Train", sign, file), os.path.join(PATH, "Validation", sign, file))


def make_debug_set(proportion_or_number = 0.2):
    # takes a random subset of the kaggle data and creates a debugging set for testing
    # the debug images are copied from the Testing data, this function does not remove
    # anything from the testing data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.getcwd()
    classes = os.listdir(os.path.join(PATH, "Test"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Debug", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Test", sign))
        random.shuffle(list_of_files)
        if proportion_or_number > 1:
            n_idxs = proportion_or_number
        else:
            n_idxs = int(len(list_of_files)*proportion_or_number)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            copyfile(os.path.join(PATH, "Test", sign, file), os.path.join(PATH, "Debug", sign, file))


def create_batch(list_of_tensors):
    input_batch = list_of_tensors.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def get_model_prediction_probs(model, input):
    # feeds an image to a neural network and returns the predictions vector
    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input)

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    sm = torch.nn.functional.softmax(output[0], dim=0)
    sm_list = sm.tolist()

    return sm_list


def train_model(model, dataloaders, num_epochs=EPOCHS, lr = 0.001):
    # trains a neural network on the dataloader data
    # this code is adapted from the PyTorch tutorial
    # at https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    since = time.time()

    val_loss_history = []
    val_acc_history = []
    training_loss_history = []
    training_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            count = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                count += 1
                if (count % 100 == 0):
                    print("Completed batch " + str(count))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Train':
                training_acc_history.append(epoch_acc)
                training_loss_history.append(epoch_loss)
            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'Validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_acc_history, val_acc_history, training_loss_history, val_loss_history


def load_and_test_model(modelpath, test_path = None, verbose = False):
    # loads a model and tests it using a dataloader and manually

    print("Loading model " + modelpath + "...")
    model = load_model(os.path.join(os.getcwd(), modelpath))
    print("Successfully loaded model.")


if __name__ == '__main__':
    # Equates to one random 28x28 image
    random_data = torch.rand((INPUT_SIZE))

    my_nn = Net()
    result = my_nn(random_data)
    print(result)
