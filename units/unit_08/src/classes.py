from copy import deepcopy

import torch
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class MonodMechanisticModel:

    def __init__(self):
        self.monod_params = None

    def fit(self, x, y):
        popt, _ = curve_fit(self.monod_equation, x, y)
        self.monod_params = popt
        return self

    def predict(self, x):
        y_pred = self.monod_equation(x, *self.monod_params)
        return y_pred

    def monod_equation(self, s, mu_max, k_s):
        mu = mu_max * (s / (k_s + s))
        return mu

    def get_mu_max(self):
        return self.monod_params[0]

    def get_k_s(self):
        return self.monod_params[1]


class MonodDataDrivenModel:

    def __init__(self, hidden_layers=(8, 8, 8, 8), activation="relu", learning_rate=0.001, epochs=1000, batch_size=1024,
                 weight_decay=0.0, shuffle=True, valid_frac=0.1, random_state=None, verbose=False):

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.verbose = verbose

        # Objects initialized later during training.
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, x, y):
        """
        Train the neural network.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input features.

        y : ndarray of shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        FullyConnectedNeuralNetwork
            Fitted estimator.
        """

        n_features = x.shape[1]
        n_targets = y.shape[1]

        # Step 1: Split data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.valid_frac,
                                                              random_state=self.random_state)

        # Step 2: Standardize inputs and outputs separately
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.scaler_x.fit(x_train)
        self.scaler_y.fit(y_train)

        x_train = self.scaler_x.transform(x_train)
        y_train = self.scaler_y.transform(y_train)
        x_valid = self.scaler_x.transform(x_valid)
        y_valid = self.scaler_y.transform(y_valid)

        # Step 3: Convert NumPy arrays to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Step 4: Create a dataloader for training data to simplify batch iteration
        dataset_train = TensorDataset(x_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

        # Step 5: Build the neural network
        self.build_model(n_features, n_targets)

        best_loss_valid = None
        best_model_state = None

        # Step 6: Train for the specified number of epochs
        for epoch in range(1, self.epochs + 1):
            loss_train = self.train_iteration(dataloader_train)
            loss_valid = self.valid_iteration(x_valid, y_valid)

            # Save the model weights whenever validation loss improves
            if (best_loss_valid is None) or (loss_valid < best_loss_valid):
                best_loss_valid = loss_valid
                best_model_state = deepcopy(self.model.state_dict())

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch: {epoch:5d} | Loss (Train): {loss_train:10.3f} | Loss (Valid): {loss_valid:10.3f}")

        # Step 7: Recreate the model and restore the weights associated with the lowest validation loss
        self.build_model(n_features, n_targets)
        self.model.load_state_dict(best_model_state)

        return self

    def predict(self, x):
        """
        Generate predictions for new input samples.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, n_targets)
            Predicted target values in the original target scale.
        """

        # Standardize the raw input data
        x = self.scaler_x.transform(x)

        # Convert the NumPy array to a PyTorch tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            y_pred = self.model(x)

        # Convert the PyTorch tensor to a NumPy array
        y_pred = y_pred.numpy(force=True)

        # Convert the standardized predictions back to the original target scale
        y_pred = self.scaler_y.inverse_transform(y_pred)

        return y_pred

    def build_model(self, n_features, n_targets):
        """
        Build the neural network architecture and optimizer.

        Parameters
        ----------
        n_features : int
            Number of input features.

        n_targets : int
            Number of output targets.
        """

        # Define the activation function
        valid_activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}
        nn_Activation = valid_activations[self.activation]

        seq_stack = []
        n_nodes_in = n_features

        # Add hidden layers with activation function
        for n_nodes_out in self.hidden_layers:
            seq_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            seq_stack.append(nn_Activation())

            n_nodes_in = n_nodes_out

        # Output layer uses a linear activation because this model is designed for regression
        seq_stack.append(nn.Linear(n_nodes_in, n_targets))

        # Create the model
        self.model = nn.Sequential(*seq_stack)

        # Define the loss function
        self.loss_fn = nn.MSELoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)

    def train_iteration(self, dataloader_train):
        """
        Perform one training epoch.

        Parameters
        ----------
        dataloader_train : DataLoader
            Mini-batches of training data.

        Returns
        -------
        float
            Mean training loss across all samples.
        """

        loss_sum = 0
        n_samples = 0

        # Switch the model to training mode
        self.model.train()

        # Iterate over the batches created by the dataloader
        for x_batch_train, y_batch_train in dataloader_train:
            # Make a prediction on one batch
            y_batch_pred = self.model(x_batch_train)

            # Calculate the loss of one batch
            loss_batch = self.loss_fn(y_batch_pred, y_batch_train)

            # Compute gradients and update network weights based on loss from this batch
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cumulate the batch losses
            batch_size = x_batch_train.shape[0]
            loss_sum += loss_batch.item() * batch_size
            n_samples += batch_size

        # Calculate the overall loss of this epoch
        loss_train = loss_sum / n_samples

        return loss_train

    def valid_iteration(self, x_valid, y_valid):
        """
        Evaluate the model on the validation dataset.

        Parameters
        ----------
        x_valid : torch.Tensor
            Validation inputs.

        y_valid : torch.Tensor
            Validation targets.

        Returns
        -------
        float
            Validation loss.
        """

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            y_pred = self.model(x_valid)
            loss = self.loss_fn(y_pred, y_valid)

        # Convert PyTorch tensor to Python float
        loss_valid = loss.item()

        return loss_valid


class MonodHybridModel:

    def __init__(self, hidden_layers=(8, 8, 8, 8), activation="relu", learning_rate=0.001, epochs=1000,
                 batch_size=1024,
                 weight_decay=0.0, shuffle=True, valid_frac=0.1, random_state=None, verbose=False):

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.verbose = verbose

        # Objects initialized later during training.
        self.model = None
        self.loss_fn = None
        self.optimizer = None

    def fit(self, x, y):
        """
        Train the neural network.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input features.

        y : ndarray of shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        FullyConnectedNeuralNetwork
            Fitted estimator.
        """

        n_features = x.shape[1]
        n_targets = y.shape[1]

        # Step 1: Split data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.valid_frac,
                                                              random_state=self.random_state)

        # Step 3: Convert NumPy arrays to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Step 4: Create a dataloader for training data to simplify batch iteration
        dataset_train = TensorDataset(x_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

        # Step 5: Build the neural network
        self.build_model(n_features, n_targets)

        best_loss_valid = None
        best_model_state = None

        # Step 6: Train for the specified number of epochs
        for epoch in range(1, self.epochs + 1):
            loss_train = self.train_iteration(dataloader_train)
            loss_valid = self.valid_iteration(x_valid, y_valid)

            # Save the model weights whenever validation loss improves
            if (best_loss_valid is None) or (loss_valid < best_loss_valid):
                best_loss_valid = loss_valid
                best_model_state = deepcopy(self.model.state_dict())

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch: {epoch:5d} | Loss (Train): {loss_train:10.3f} | Loss (Valid): {loss_valid:10.3f}")

        # Step 7: Recreate the model and restore the weights associated with the lowest validation loss
        self.build_model(n_features, n_targets)
        self.model.load_state_dict(best_model_state)

        return self

    def predict(self, x):
        """
        Generate predictions for new input samples.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, n_targets)
            Predicted target values in the original target scale.
        """

        # Convert the NumPy array to a PyTorch tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            y_pred = self.model(x)

        # Convert the PyTorch tensor to a NumPy array
        y_pred = y_pred.numpy(force=True)

        return y_pred

    def build_model(self, n_features, n_targets):
        """
        Build the neural network architecture and optimizer.

        Parameters
        ----------
        n_features : int
            Number of input features.

        n_targets : int
            Number of output targets.
        """

        # Create the model
        self.model = MonodHybridModelPyTorch(hidden_layers=self.hidden_layers, activation=self.activation)

        # Define the loss function
        self.loss_fn = nn.MSELoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)

    def train_iteration(self, dataloader_train):
        """
        Perform one training epoch.

        Parameters
        ----------
        dataloader_train : DataLoader
            Mini-batches of training data.

        Returns
        -------
        float
            Mean training loss across all samples.
        """

        loss_sum = 0
        n_samples = 0

        # Switch the model to training mode
        self.model.train()

        # Iterate over the batches created by the dataloader
        for x_batch_train, y_batch_train in dataloader_train:
            # Make a prediction on one batch
            y_batch_pred = self.model(x_batch_train)

            # Calculate the loss of one batch
            loss_batch = self.loss_fn(y_batch_pred, y_batch_train)

            # Compute gradients and update network weights based on loss from this batch
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cumulate the batch losses
            batch_size = x_batch_train.shape[0]
            loss_sum += loss_batch.item() * batch_size
            n_samples += batch_size

        # Calculate the overall loss of this epoch
        loss_train = loss_sum / n_samples

        return loss_train

    def valid_iteration(self, x_valid, y_valid):
        """
        Evaluate the model on the validation dataset.

        Parameters
        ----------
        x_valid : torch.Tensor
            Validation inputs.

        y_valid : torch.Tensor
            Validation targets.

        Returns
        -------
        float
            Validation loss.
        """

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            y_pred = self.model(x_valid)
            loss = self.loss_fn(y_pred, y_valid)

        # Convert PyTorch tensor to Python float
        loss_valid = loss.item()

        return loss_valid


class MonodHybridModelPyTorch(nn.Module):

    def __init__(self, hidden_layers, activation, eps=(10 ** (-6))):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.eps = eps

        beta=5
        hidden_layers_2 = (4,4)

        # Define the activation function
        valid_activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}
        nn_Activation = valid_activations[self.activation]

        base_stack = []
        n_nodes_in = 2  # pH, T

        # Add hidden layers with activation function
        for n_nodes_out in self.hidden_layers:
            base_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            base_stack.append(nn_Activation())

            n_nodes_in = n_nodes_out

        # Output layer uses a linear activation because this model is designed for regression

        n_nodes_in = self.hidden_layers[-1]
        mu_max_stack = []
        for n_nodes_out in hidden_layers_2:
            mu_max_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            mu_max_stack.append(nn_Activation())
            n_nodes_in = n_nodes_out
        mu_max_stack.append(nn.Linear(n_nodes_in, 1))
        mu_max_stack.append(nn.Softplus(beta=beta))

        n_nodes_in = self.hidden_layers[-1]
        k_s_stack = []
        for n_nodes_out in hidden_layers_2:
            k_s_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            k_s_stack.append(nn_Activation())
            n_nodes_in = n_nodes_out
        k_s_stack.append(nn.Linear(n_nodes_in, 1))
        k_s_stack.append(nn.Softplus(beta=beta))

        self.layers_base = nn.Sequential(*base_stack)
        self.layers_mu_max = nn.Sequential(*mu_max_stack)
        self.layers_k_s = nn.Sequential(*k_s_stack)

    def forward(self, x):
        # x = [s, ph, temp]

        s = x[:, :1]
        pt = x[:, 1:]

        base = self.layers_base(pt)
        mu_max = self.layers_mu_max(base)
        k_s = self.layers_k_s(base)

        mu = mu_max * (s / (k_s + s + self.eps))

        return mu
