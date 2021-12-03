from os import XATTR_SIZE_MAX
import torch 
import torch.nn as nn
import scipy.io as sp
import sklearn.metrics
import numpy as np
import sys
import matplotlib.pyplot as plt

def plot_data(X, Y):

    data_x_class_pos = []
    data_y_class_pos = []
    data_x_class_neg = []
    data_y_class_neg = []

    for i in range(0, len(X)):
        if Y[i] == -1:
            data_x_class_pos.append(X[i][0]) 
            data_y_class_pos.append(X[i][1])
        else:
            data_x_class_neg.append(X[i][0]) 
            data_y_class_neg.append(X[i][1])

    plt.scatter(data_x_class_pos, data_y_class_pos, color='orange')
    plt.scatter(data_x_class_neg, data_y_class_neg, color='blue')
    plt.show()

def converge_to_binary(x):
    x = np.where(x < .5, -1, 1)
    return x

def converge_to_prob(x):
    x = np.where(x < 0., 0., 1.)
    return x

data = sp.loadmat("hw04_dataset.mat")

# print(data)

X = data["X_trn"]
Y = data["y_trn"]

# print(X)

X_test = data["X_tst"]
Y_test = data["y_tst"]

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 2, 50, 1, 2, 'relu'

x = torch.tensor(X).float()
y = torch.tensor(Y).float()

x_test = torch.tensor(X_test).float()
y_test = torch.tensor(Y_test).float()


class MyModule(nn.Module):
    def __init__(self, n_in, neurons_per_hidden, n_out, hidden_layers, activation_function):
        super(MyModule, self).__init__()

        self.n_in = n_in
        self.n_h = neurons_per_hidden
        self.n_out = n_out
        self.h_l = hidden_layers

        self.a_f = activation_function

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_in, self.n_h)
        # Defaults to Relu if activation_function is improperly sp
        self.activation_layer = nn.ReLU()

        if activation_function == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation_function == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation_function == 'sigmoid':
            self.activation_layer == nn.Sigmoid()
        elif activation_function == 'identity':
            self.activation_layer == nn.Identity()
        else:
            print("Invalid activation function specified")
            sys.exit(1)

        self.linears = nn.ModuleList([nn.Linear(self.n_h, self.n_h) for i in range(self.h_l - 1)])
        self.activation_layers = nn.ModuleList([self.activation_layer for i in range(self.h_l - 1)])
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(self.n_h, n_out)
        # Define sigmoid output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)
        x = self.activation_layer(x)

        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
            x = self.activation_layers[i // 2](x) + l(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x

model = nn.Sequential(MyModule(n_in, n_h, n_out, hidden_layers, activation_function))
# print(model)

# Construct the loss function
criterion = torch.nn.BCELoss()
# Construct the optimizer (Adam in this case)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Optimization
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   y_pred_numpy = y_pred.detach().numpy()
   y_pred_tanh_range = converge_to_binary(y_pred_numpy)
   y_numpy = y.detach().numpy()
   y_sigmoid_range = converge_to_prob(y_numpy)

#    print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y))

   # Compute and print loss
   loss = criterion(y_pred, torch.tensor(y_sigmoid_range).float())
#    print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()

# Plotting training data
# plot_data(X, Y)

# Printing all weights and biases from NN

print("All of the weights and biases from all layers in model")

for i in model.named_modules():
    print(i[0])

    for j in i[1].state_dict().items():
        print(j)

y_pred_tanh_range_from_train = y_pred_tanh_range

y_pred = model(x_test)
y_pred_numpy = y_pred.detach().numpy()
y_pred_tanh_range = converge_to_binary(y_pred_numpy)

print("Results of last activation layer")
print(y_pred)

print("Training Accuracy")
print(sklearn.metrics.accuracy_score(y_pred_tanh_range_from_train, y))

print("Testing Accuracy")
print(sklearn.metrics.accuracy_score(y_pred_tanh_range, y_test))

# Plotting the test data
# plot_data(x_test, y_pred_tanh_range)

