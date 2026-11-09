from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMNeuralNetwork:

    def __init__(self, nodes_per_layer, n_layers, learning_rate=0.001, epochs=1000, batch_size=1024,
                 weight_decay=0.0, shuffle=True, valid_frac=0.1, random_state=None, verbose=False):

        self.nodes_per_layer = nodes_per_layer
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.shuffle = shuffle
        self.valid_frac = valid_frac
        self.random_state = random_state
        self.verbose = verbose

        self.rng = np.random.default_rng(0)

        # Objects initialized later during training.
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scaler_mean = None
        self.scaler_sd = None
        self.seq_length = None

    def fit(self, x, y, group_ids):

        n_features = x.shape[2]

        # Step 1: Split data into training and validation sets
        u_group_ids = np.unique(group_ids)
        n_groups = u_group_ids.size
        n_groups_valid = round(n_groups * self.valid_frac)
        group_ids_valid = set(self.rng.choice(u_group_ids, n_groups_valid, replace=False))

        mask_valid = np.array([(group_id in group_ids_valid) for group_id in group_ids])
        mask_train = ~mask_valid

        x_train = x[mask_train]
        y_train = y[mask_train]

        x_valid = x[mask_valid]
        y_valid = y[mask_valid]

        # Step 2: Standardize data
        scaler_train = np.concatenate((x_train.reshape(-1, n_features), y_train.reshape(-1, n_features)), axis=0)
        scaler_train = np.unique(scaler_train, axis=0)

        self.scaler_mean = np.mean(scaler_train, axis=0)
        self.scaler_sd = np.std(scaler_train, axis=0)

        scaler_m = np.reshape(self.scaler_mean, (1, 1, n_features))
        scaler_s = np.reshape(self.scaler_sd, (1, 1, n_features))

        x_train = (x_train - scaler_m) / scaler_s
        y_train = (y_train - scaler_m) / scaler_s
        x_valid = (x_valid - scaler_m) / scaler_s
        y_valid = (y_valid - scaler_m) / scaler_s

        # Step 3: Convert NumPy arrays to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_valid = torch.tensor(x_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Step 4: Create dataloaders
        dataset_train = TensorDataset(x_train, y_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

        dataset_valid = TensorDataset(x_valid, y_valid)
        dataloader_valid = DataLoader(dataset_valid, batch_size=self.batch_size, shuffle=self.shuffle)

        # Step 5: Build the neural network
        self.build_model(n_features)

        best_loss_valid = None
        best_model_state = None

        # Step 6: Train for the specified number of epochs
        for epoch in range(1, self.epochs + 1):
            loss_train = self.train_iteration(dataloader_train)
            loss_valid = self.valid_iteration(dataloader_valid)

            # Save the model weights whenever validation loss improves
            if (best_loss_valid is None) or (loss_valid < best_loss_valid):
                best_loss_valid = loss_valid
                best_model_state = deepcopy(self.model.state_dict())

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch: {epoch:5d} | Loss (Train): {loss_train:10.3f} | Loss (Valid): {loss_valid:10.3f}")

        # Step 7: Recreate the model and restore the weights associated with the lowest validation loss
        self.build_model(n_features)
        self.model.load_state_dict(best_model_state)

        return self

    def predict(self, x, hidden_state_0=None):
        n_features = x.shape[-1]

        if len(x.shape) == 2:
            scaler_m = np.reshape(self.scaler_mean, (1, n_features))
            scaler_s = np.reshape(self.scaler_sd, (1, n_features))
        elif len(x.shape) == 3:
            scaler_m = np.reshape(self.scaler_mean, (1, 1, n_features))
            scaler_s = np.reshape(self.scaler_sd, (1, 1, n_features))
        else:
            scaler_m = None
            scaler_s = None

        # Standardize the raw input data
        x = (x - scaler_m) / scaler_s

        # Convert the NumPy array to a PyTorch tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            if hidden_state_0 is not None:
                y_pred, hidden_state_n = self.model(x, hidden_state_0)
            else:
                y_pred, hidden_state_n = self.model(x)

        # Convert the PyTorch tensor to a NumPy array
        y_pred = y_pred.numpy(force=True)

        # Convert the standardized predictions back to the original target scale
        y_pred = y_pred * scaler_s + scaler_m

        return y_pred, hidden_state_n

    def build_model(self, n_features):
        # Create the model
        self.model = LSTMNeuralNetworkPyTorch(input_size=n_features, output_size=n_features,
                                              hidden_size=self.nodes_per_layer, num_layers=self.n_layers)

        # Define the loss function
        self.loss_fn = nn.MSELoss()

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                           weight_decay=self.weight_decay)

    def train_iteration(self, dataloader_train):
        loss_sum = 0
        n_samples = 0

        # Switch the model to training mode
        self.model.train()

        # Iterate over the batches created by the dataloader
        for x_batch_train, y_batch_train in dataloader_train:
            # Make a prediction on one batch
            y_batch_pred, _ = self.model(x_batch_train)

            # Calculate the loss of one batch
            loss_batch = self.loss_fn(y_batch_pred, y_batch_train)

            # Compute gradients and update network weights based on loss from this batch
            loss_batch.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Cumulate the batch losses
            batch_size = x_batch_train.shape[0] * x_batch_train.shape[1]
            loss_sum += loss_batch.item() * batch_size
            n_samples += batch_size

        # Calculate the overall loss of this epoch
        loss_train = loss_sum / n_samples

        return loss_train

    def valid_iteration(self, dataloader_valid):
        loss_sum = 0
        n_samples = 0

        # Switch the model to evaluation mode
        self.model.eval()

        # Disable gradient tracking during prediction
        with torch.no_grad():
            # Iterate over the batches created by the dataloader
            for x_batch_valid, y_batch_valid in dataloader_valid:
                # Make a prediction on one batch
                y_batch_pred, _ = self.model(x_batch_valid)

                # Calculate the loss of one batch
                loss_batch = self.loss_fn(y_batch_pred, y_batch_valid)

                # Cumulate the batch losses
                batch_size = x_batch_valid.shape[0] * x_batch_valid.shape[1]
                loss_sum += loss_batch.item() * batch_size
                n_samples += batch_size

        # Calculate the overall loss of this epoch
        loss_valid = loss_sum / n_samples

        return loss_valid


class LSTMNeuralNetworkPyTorch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            x, (h_n, c_n) = self.lstm(x)
        else:
            x, (h_n, c_n) = self.lstm(x, hidden_state)

        x = self.linear(x)
        return x, (h_n, c_n)
