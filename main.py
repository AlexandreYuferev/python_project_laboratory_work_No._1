
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

np.random.seed(0)

matplotlib.rcParams['figure.figsize'] = (12.5, 5.0)


def sig(x):
    return 1/(1 + np.exp(-x))


def dev_sig(x):
    return sig(x) * (1 - sig(x))


class FullyConnectedNN:

    def __init__(self):

        self.input_dim    = 1
        self.output_dim   = 1
        self.hidden_dim_1 = 25
        self.hidden_dim_2 = 25

        self.W1 = np.random.uniform(-1, 1, size=(self.hidden_dim_1, self.input_dim))
        self.W2 = np.random.uniform(-1, 1, size=(self.hidden_dim_2, self.hidden_dim_1))
        self.W3 = np.random.uniform(-1, 1, size=(self.output_dim  , self.hidden_dim_2))

        self.B1 = np.random.uniform(-1, 1, size=(self.hidden_dim_1, 1))
        self.B2 = np.random.uniform(-1, 1, size=(self.hidden_dim_2, 1))
        self.B3 = np.random.uniform(-1, 1, size=(self.output_dim  , 1))

        self.Y1 = self.S1 = np.zeros((self.hidden_dim_1, 1))
        self.Y2 = self.S2 = np.zeros((self.hidden_dim_2, 1))
        self.Y3 = self.S3 = np.zeros((self.output_dim  , 1))

        self.error = None
        self.error_derivative = None

        """
        print('W1 = \n{0}\n'.format(self.W1))
        print('S1 = \n{0}\n'.format(self.S1))
        print('W2 = \n{0}\n'.format(self.W2))
        print('S2 = \n{0}\n'.format(self.S2))
        print('W3 = \n{0}\n'.format(self.W3))
        print('S3 = \n{0}\n'.format(self.S3))
        # """

    def forward_pass(self, input_array):

        output_array = []

        for X in input_array:

            # (hidden_dim_1, input_dim) @ (1, input_dim).T --> (hidden_dim_1, 1)
            self.S1 = self.B1 + self.W1 * X
            self.Y1 = sig(self.S1)

            """
            print(f'W1 = \n{self.W1}\n')
            print(f'S1 = \n{self.S1}\n')
            print(f'Y1 = \n{self.Y1}\n')
            print('-'*32)
            # """

            # (hidden_dim_2, hidden_dim_1) @ (hidden_dim_1, 1) --> (hidden_dim_2, 1)
            self.S2 = self.B2 + self.W2.dot(self.Y1)
            self.Y2 = sig(self.S2)

            """
            print(f'W2 = \n{self.W2}\n')
            print(f'S2 = \n{self.S2}\n')
            print(f'Y2 = \n{self.Y2}\n')
            print('-' * 32)
            # """

            # (output_dim, hidden_dim_2) @ (hidden_dim_2, 1) --> (output_dim, 1)
            self.S3 = self.B3 + self.W3.dot(self.Y2)
            self.Y3 = sig(self.S3)

            """
            print(f'W3 = \n{self.W3}\n')
            print(f'S3 = \n{self.S3}\n')
            print(f'Y3 = \n{self.Y3}\n')
            print('-' * 32)
            # """

            output_array.append(self.Y3[0])

        return np.asarray(output_array)

    def backward_pass(self, input_array):

        delta_3 = self.error * dev_sig(self.S3)
        dW3 = delta_3 * self.Y2.T

        """
        print(f'delta_3 = \n{delta_3}\n')
        print(f'dW3 = \n{dW3}\n')
        # """

        delta_2 = (self.W3.T * delta_3) * dev_sig(self.S2)
        dW2 = delta_2 * self.Y1.T

        """
        print(f'delta_2 = \n{delta_2}\n')
        print(f'dW2 = \n{dW2}\n')
        # """

        delta_1 = (self.W2.T @ delta_2) * dev_sig(self.S1)
        dW1 = delta_1 * input_array[random.randint(0, 999)]

        """
        print(f'delta_1 = \n{delta_1}\n')
        print(f'dW1 = \n{dW1}\n')
        # """

        self.W1 -= 0.25 * dW1
        self.W2 -= 0.25 * dW2
        self.W3 -= 0.25 * dW3

    def calculate_loss(self, output_array, target_array):
        self.error = ((target_array - output_array) ** 2).mean()
        self.error_derivative = (output_array - target_array).mean()


if __name__ == '__main__':

    x_train = np.random.rand(1000) * 20.0 - 10.0
    y_train = 0.8 * np.sin(x_train) + np.random.uniform(-0.195, 0.195, size=(len(x_train)))

    neural_network = FullyConnectedNN()

    loss_array = []
    predictions = []
    for epoch_index in range(1000):
        predictions = neural_network.forward_pass(np.expand_dims(x_train, axis=1))
        neural_network.calculate_loss(predictions, np.expand_dims(y_train, axis=1))
        neural_network.backward_pass(np.expand_dims(x_train, axis=1))
        loss_array.append(neural_network.error)
        print(f'epoch index = {epoch_index}')

    fig, ax = plt.subplots()
    ax.plot(loss_array, 'o', markersize=2)
    ax.grid()
    plt.show()

    _, ax = plt.subplots()
    ax.plot(x_train, y_train, 'o', markersize=2)
    ax.plot(x_train, predictions, 'o', markersize=2)
    ax.grid()
    plt.show()







