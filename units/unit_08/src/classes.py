from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class MonodMechanisticModel:
    """
    Mechanistic growth model based on Monod kinetics with environmental
    correction terms for pH and temperature.
    """

    def __init__(self):
        self.monod_params = None

    def fit(self, x, y):
        """
        Fit the mechanistic Monod model to experimental data.

        Parameters
        ----------
        x : ndarray of shape (n_samples, 3)
            Input features arranged as:

            - x[:, 0] = substrate concentration (S)
            - x[:, 1] = pH
            - x[:, 2] = temperature

        y : ndarray of shape (n_samples, 1)
            Measured specific growth rates.

        Returns
        -------
        MonodMechanisticModel
            Fitted estimator.
        """

        # Small positive lower bound used to prevent parameters from becoming zero or negative during optimization.
        eps = 10 ** (-6)

        # Convert inputs to the format expected by scipy.curve_fit.
        x_cf = self.process_x(x)
        y_cf = self.process_y(y)

        # Initial parameter guesses.
        # Good initial values help the nonlinear optimizer converge to a physically meaningful solution.
        k_s_opt_0 = 10
        mu_opt_0 = 1
        ph_opt = 7.4
        ph_bandwidth = 0.1 * ph_opt
        temp_opt = 37
        temp_bandwidth = 0.1 * temp_opt
        p0 = [k_s_opt_0, mu_opt_0, ph_opt, ph_bandwidth, temp_opt, temp_bandwidth]

        # Constrain all parameters to remain strictly positive.
        bounds = (eps, np.inf)

        # Estimate parameters by minimizing the squared error between measured and predicted growth rates.
        popt, _ = curve_fit(self.monod_equation, x_cf, y_cf, p0=p0, bounds=bounds)
        self.monod_params = popt

        return self

    def predict(self, x):
        """
        Predict growth rates for new operating conditions.

        Parameters
        ----------
        x : ndarray of shape (n_samples, 3)
            Input features arranged as:

            - x[:, 0] = substrate concentration (S)
            - x[:, 1] = pH
            - x[:, 2] = temperature

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Predicted specific growth rates.
        """

        # Convert the design matrix into the tuple format expected by the mechanistic equation.
        x_cf = self.process_x(x)

        # Evaluate the fitted Monod equation.
        y_pred = self.monod_equation(x_cf, *self.monod_params)

        # Return predictions as a column vector to match the target format.
        y_pred = np.expand_dims(y_pred, axis=1)

        return y_pred

    def monod_equation(self, x, k_s_opt, mu_opt, ph_opt, ph_bandwidth, temp_opt, temp_bandwidth):
        """
        Evaluate the environmentally corrected Monod equation.

        Parameters
        ----------
        x : tuple of ndarray
            Tuple containing:

            - substrate concentration (S)
            - pH
            - temperature

        k_s_opt : float
            Half-saturation constant under optimal environmental conditions.

        mu_opt : float
            Maximum specific growth rate under optimal environmental
            conditions.

        ph_opt : float
            Optimal pH.

        ph_bandwidth : float
            Controls the sensitivity of growth to deviations from the
            optimal pH.

        temp_opt : float
            Optimal temperature.

        temp_bandwidth : float
            Controls the sensitivity of growth to deviations from the
            optimal temperature.

        Returns
        -------
        ndarray
            Predicted specific growth rates.
        """

        s, ph, temp = x

        # Normalized squared distance from the optimal pH.
        a_ph = ((ph - ph_opt) / ph_bandwidth) ** 2

        # Normalized squared distance from the optimal temperature.
        a_temp = ((temp - temp_opt) / temp_bandwidth) ** 2

        # Combined environmental penalty.
        # a_env = 0 at the optimal conditions and increases as pH and/or
        # temperature move away from their optimal values.
        a_env = a_ph + a_temp

        # Effective maximum growth rate.
        # At the optimal conditions, exp(-a_env) = 1 and mu_max = mu_opt.
        # Far from the optimum, exp(-a_env) approaches zero.
        mu_max = mu_opt * np.exp(-a_env)

        # Effective half-saturation constant.
        # At the optimal conditions, exp(a_env) = 1 and k_s = k_s_opt.
        # Far from the optimum, k_s approaches infinity.
        k_s = k_s_opt * np.exp(a_env)

        # Monod equation.
        mu = mu_max * (s / (k_s + s))

        return mu

    def process_x(self, x):
        """
        Split the feature matrix into individual model inputs.

        Parameters
        ----------
        x : ndarray of shape (n_samples, 3)
            Input feature matrix.

        Returns
        -------
        tuple of ndarray
            Tuple containing:

            - substrate concentration (S)
            - pH
            - temperature
        """
        s = x[:, 0]
        ph = x[:, 1]
        temp = x[:, 2]

        return s, ph, temp

    def process_y(self, y):
        """
        Convert the target matrix into a one-dimensional response vector.

        Parameters
        ----------
        y : ndarray of shape (n_samples, 1)
            Target values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Flattened target vector suitable for use with curve_fit.
        """

        return y[:, 0]


class MonodDataDrivenModel:
    """
    Fully data-driven neural network model for predicting cell growth.

    This model does not impose any Monod kinetics (despite the class name).
    Instead, it learns the relationship between the input variables and
    target response directly from data using a fully connected neural network.

    Parameters
    ----------
    hidden_layers : tuple of int, default=(16, 16, 16)
        Number of neurons in each hidden layer.

    activation : {"relu", "leakyrelu", "tanh", "sigmoid"}, default="relu"
        Activation function used in all hidden layers.

    learning_rate : float, default=0.001
        Learning rate used by the AdamW optimizer.

    epochs : int, default=200
        Number of training epochs.

    batch_size : int, default=1024
        Number of samples per training batch.

    weight_decay : float, default=0.0
        Weight decay coefficient used by AdamW.

    shuffle : bool, default=True
        Whether training samples should be shuffled before each epoch.

    valid_frac : float, default=0.1
        Fraction of data reserved for validation.

    random_state : int or None, default=None
        Random seed used when splitting training and validation data.

    verbose : bool, default=False
        If True, training and validation losses are periodically displayed.
    """

    def __init__(self, hidden_layers=(16, 16, 16), activation="relu", learning_rate=0.001, epochs=200, batch_size=1024,
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
    """
    Hybrid mechanistic-machine learning model for predicting cell growth.

    This model combines a neural network with the Monod growth equation.
    Instead of directly predicting the growth rate, the neural network
    learns how environmental conditions influence the Monod parameters:

    - mu_max : maximum specific growth rate
    - k_s : half-saturation constant

    The neural network receives pH and temperature as inputs and predicts
    values of mu_max and k_s. These learned parameters are then used in
    the Monod equation:

        mu = mu_max * s / (k_s + s)

    where s is the substrate concentration.

    Parameters
    ----------
    hidden_layers : tuple of int, default=(16, 16, 16)
        Number of neurons in each hidden layer.

    activation : {"relu", "leakyrelu", "tanh", "sigmoid"}, default="relu"
        Activation function used in all hidden layers.

    learning_rate : float, default=0.001
        Learning rate used by the AdamW optimizer.

    epochs : int, default=200
        Number of training epochs.

    batch_size : int, default=1024
        Number of samples per training batch.

    weight_decay : float, default=0.0
        Weight decay coefficient used by AdamW.

    shuffle : bool, default=True
        Whether training samples should be shuffled before each epoch.

    valid_frac : float, default=0.1
        Fraction of data reserved for validation.

    random_state : int or None, default=None
        Random seed used when splitting training and validation data.

    verbose : bool, default=False
        If True, training and validation losses are periodically displayed.
    """

    def __init__(self, hidden_layers=(16, 16, 16), activation="relu", learning_rate=0.001, epochs=200, batch_size=1024,
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
        self.scaler_pt = None

    def fit(self, x, y):
        """
        Fit the hybrid Monod model to experimental data.

        Parameters
        ----------
        x : ndarray of shape (n_samples, 3)
            Input features arranged as:

            - x[:, 0] = substrate concentration (S)
            - x[:, 1] = pH
            - x[:, 2] = temperature

        y : ndarray of shape (n_samples, 1)
            Measured specific growth rates.

        Returns
        -------
        MonodHybridModel
            Fitted estimator.
        """

        # Create copies to avoid in-place modification
        x = np.copy(x)
        y = np.copy(y)

        # Step 1: Split data into training and validation sets
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.valid_frac,
                                                              random_state=self.random_state)

        # Step 2: Standardize pH and temperature only, since they are the inputs to the neural network.
        #         S needs to appear as-is in the Monod equation.
        self.scaler_pt = StandardScaler()
        self.scaler_pt.fit(x_train[:, 1:])

        # Note: If we didn't make copies earlier, this step would have modified the original arrays!
        x_train[:, 1:] = self.scaler_pt.transform(x_train[:, 1:])
        x_valid[:, 1:] = self.scaler_pt.transform(x_valid[:, 1:])

        # Step 3: Convert NumPy arrays to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Step 4: Create a dataloader for training data to simplify batch iteration
        dataset_train = TensorDataset(x_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

        # Step 5: Build the neural network
        self.build_model()

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
        self.build_model()
        self.model.load_state_dict(best_model_state)

        return self

    def predict(self, x):
        """
        Predict growth rates for new operating conditions.

        Parameters
        ----------
        x : ndarray of shape (n_samples, 3)
            Input features arranged as:

            - x[:, 0] = substrate concentration (S)
            - x[:, 1] = pH
            - x[:, 2] = temperature

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Predicted specific growth rates.
        """

        # Create copies to avoid in-place modification
        x = np.copy(x)

        # Standardize pH and temperature only, since they are the inputs to the neural network.
        x[:, 1:] = self.scaler_pt.transform(x[:, 1:])

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

    def build_model(self):
        """
        Build the neural network architecture and optimizer.
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
    """
    PyTorch implementation of the hybrid Monod model.

    The model decomposes cell growth prediction into two parts:

    1. A neural network learns how pH and temperature affect
       the Monod parameters.

    2. A mechanistic Monod equation converts the learned
       parameters and substrate concentration into a specific
       growth rate.

    Parameters
    ----------
    hidden_layers : tuple of int
        Number of neurons in each hidden layer.

    activation : {"relu", "leakyrelu", "tanh", "sigmoid"}
        Activation function used in all hidden layers.

    eps : float, default=(10 ** (-6))
        A small constant for numerical stability (i.e., avoid divide by zero scenarios).

    """

    def __init__(self, hidden_layers, activation, eps=(10 ** (-6))):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.activation = activation
        self.eps = eps

        # Define the activation function
        valid_activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU}
        nn_Activation = valid_activations[self.activation]

        # Only pH and temperature are supplied to the neural network.
        # Substrate concentration is handled separately by the mechanistic
        # Monod equation.
        n_nodes_in = 2

        base_stack = []
        # Add hidden layers with activation function
        for n_nodes_out in self.hidden_layers:
            base_stack.append(nn.Linear(n_nodes_in, n_nodes_out))
            base_stack.append(nn_Activation())

            n_nodes_in = n_nodes_out

        # Predict the environment-dependent maximum growth rate.
        # Softplus ensures non-negative outputs.
        mu_max_stack = [
            nn.Linear(n_nodes_in, 1),
            nn.Softplus()
        ]

        # Predict the environment-dependent half-saturation constant.
        # Softplus ensures non-negative outputs.
        k_s_stack = [
            nn.Linear(n_nodes_in, 1),
            nn.Softplus()
        ]

        # Shared environmental representation.
        self.layers_base = nn.Sequential(*base_stack)

        # Monod parameter prediction heads.
        self.layers_mu_max = nn.Sequential(*mu_max_stack)
        self.layers_k_s = nn.Sequential(*k_s_stack)

    def forward(self, x):
        """
        Evaluate the hybrid Monod model.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, 3)
            Input matrix containing:

            - x[:, 0] = substrate concentration(S)
            - x[:, 1] = pH
            - x[:, 2] = temperature

        Returns
        -------
        torch.Tensor of shape (n_samples, 1)
            Predicted specific growth rates.
        """

        s = x[:, :1]
        pt = x[:, 1:]

        # Generate a latent representation of the environmental state (e.g., final hidden layer output)
        base = self.layers_base(pt)

        # Predict mu_max.
        mu_max = self.layers_mu_max(base)

        # Predict k_s. Add self.eps as a small positive constant to improve numerical stability.
        k_s = self.layers_k_s(base) + self.eps

        # Monod equation.
        mu = mu_max * (s / (k_s + s))

        return mu
