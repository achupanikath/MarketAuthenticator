import copy
import torch.nn.functional as F
import random
import numpy as np
from shutil import copyfile
from neural_network_helper import prep_train_val_data
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt


BATCH_SIZE = 32
INPUT_SIZE = 11
HIDDEN_LAYER_SIZE = INPUT_SIZE*4
NUM_CLASSES = 2
EPOCHS = 20


class Net(nn.Module):
    # adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def train(epochs = 50, lr = 0.01):
    #adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    csv_path = os.path.join(os.getcwd(), "ML", "all_data.csv")
    df = pd.read_csv(csv_path)
    X = df[['Epiry', 'Registration', 'Updated', 'Blacklisted', 'ServerLocation',
            'BL_Score', 'PageRank', 'HTTPS', 'Cert_recieved', 'Cert_chain', 'Cert_hostname']]
    y = df[['Phishing']]

    # split and format data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.from_numpy(X_train.to_numpy()).float()
    y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
    X_test = torch.from_numpy(X_test.to_numpy()).float()
    y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

    since = time.time()
    test_loss_history = []
    test_acc_history = []
    training_loss_history = []
    training_acc_history = []

    model = Net()
    criterion = nn.BCELoss()
    #optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        if(epochs % 25 == 0 and epoch != 0):
            lr = lr/2
        y_pred = model(X_train)
        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train)
        training_loss_history.append(train_loss)
        train_acc = calculate_accuracy(y_train, y_pred)
        training_acc_history.append(train_acc)
        y_test_pred = model(X_test)
        y_test_pred = torch.squeeze(y_test_pred)
        test_loss = criterion(y_test_pred, y_test)
        test_loss_history.append(test_loss)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        test_acc_history.append(test_acc)
        print("epoch {}\nTrain set - loss: {}, accuracy: {}\nTest  set - loss: {}, accuracy: {}\n"
              .format(str(epoch), str(train_loss), str(train_acc), str(test_loss), str(test_acc)))
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, training_acc_history, test_acc_history, training_loss_history, test_loss_history


def calculate_accuracy(y_true, y_pred):
    # adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    a = y_pred.ge(.5)
    b = a.view(-1)
    predicted = b
    return (y_true == predicted).sum().float()/len(y_true)


def get_model_prediction(model, data):
    # adapted from https://curiousily.com/posts/build-your-first-neural-network-with-pytorch/
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.as_tensor(data).float().to(device)
    output = model(t)
    return output.ge(0.5).item(), output


def save_model(model, path="pytorch_saved"):
    #save a model
    path = os.path.join(os.getcwd(), "ML", path)
    torch.save(model, path)
    print('Model saved as {}'.format(path))
    return path


def load_model():
    # load a model from a file
    filename = os.path.join(os.getcwd(), "ML", "pytorch_saved")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(filename, map_location=device)
    model.eval()
    return model


def create_train_save_print_model():
    model, training_acc, test_acc, training_loss, test_loss = train()
    save_model(model)
    plot_accuracy_and_loss(training_acc, training_loss, test_acc, test_loss)


def plot_accuracy_and_loss(train_acc, train_loss, test_acc, test_loss):
    # plots the training and validation accuracy and loss during training
    # plotting graphs for accuracy
    plt.figure(0)
    plt.plot(train_acc, label='training accuracy')
    plt.plot(test_acc, label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    path = os.path.join(os.getcwd(), "ML", "Outputs", "accuracy.png")
    plt.savefig(path)

    plt.figure(1)
    plt.plot(train_loss, label='training loss')
    plt.plot(test_loss, label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    path = os.path.join(os.getcwd(), "ML", "Outputs", "loss.png")
    plt.savefig(path)


def make_val_set(p = 0.2):
    # takes a random subset of the kaggle data and creates a validation set
    # the validation images are removed from the training data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.path.join(os.getcwd(), "ML")
    classes = os.listdir(os.path.join(PATH, "Train"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Validation", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Train", sign))
        random.shuffle(list_of_files)
        n_idxs = int(len(list_of_files)*p)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            os.rename(os.path.join(PATH, "Train", sign, file), os.path.join(PATH, "Validation", sign, file))


def make_testing_set(p = 0.2):
    # takes a random subset of the kaggle data and creates a testing set
    # the testing images are removed from the training data
    # code adapted from
    # https://medium.com/@rasmus1610/how-to-create-a-validation-set-for-image-classification-35d3ef0f47d3
    PATH = os.path.join(os.getcwd(), "ML")
    classes = os.listdir(os.path.join(PATH, "Train"))
    for sign in classes:
        os.makedirs(os.path.join(PATH, "Test", sign), exist_ok=True)
        list_of_files = os.listdir(os.path.join(PATH, "Train", sign))
        random.shuffle(list_of_files)
        n_idxs = int(len(list_of_files)*p)
        selected_files = [list_of_files[n] for n in range(n_idxs)]
        for file in selected_files:
            os.rename(os.path.join(PATH, "Train", sign, file), os.path.join(PATH, "Test", sign, file))


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


if __name__ == '__main__':

    create_train_save_print_model()