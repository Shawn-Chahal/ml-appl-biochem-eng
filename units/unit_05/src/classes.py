import numpy as np


class StandardScalerAR:
    """
    Standard scaler designed specifically for autoregressive (AR) and
    time series forecasting workflows.

    Unlike sklearn's StandardScaler, which computes a separate mean and
    standard deviation for each feature (column), this scaler computes a
    single global mean and standard deviation across the entire input array.

    This behavior is useful when autoregressive lag features all represent
    the same underlying variable at different time offsets. Scaling every
    lag using a common distribution preserves the relative relationships
    between lags and ensures all lagged values are normalized using the
    same reference statistics.

    Example
    -------
    Given lagged features:

        [[100,  95,  90],
         [105, 100,  95],
         [110, 105, 100]]

    sklearn.StandardScaler would compute statistics independently for each
    column (x_(t-1), x_(t-2), x_(t-3)).

    StandardScalerAR instead computes:

        mean = mean(all values)
        std  = std(all values)

    and applies those same statistics to every element in the array.

    Attributes
    ----------
    mean_ : float
        Global mean of all values observed during fitting.

    scale_ : float
        Global standard deviation of all values observed during fitting.

    var_ : float
        Global variance of all values observed during fitting.

    n_samples_seen_ : int
        Number of rows observed during fitting.

    n_features_in_ : int
        Number of features (columns) in the input array.

    feature_names_in_ : array-like or None
        Placeholder attribute for sklearn API compatibility.
    """

    def __init__(self):
        """Initialize an unfitted scaler."""

        self.mean_ = None
        self.scale_ = None
        self.var_ = None

        self.n_samples_seen_ = None
        self.n_features_in_ = None

        self.feature_names_in_ = None

    def fit(self, x, y=None):
        """
        Compute global scaling statistics from the input data.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features)
            Input training data.

        y : ignored, default=None
            Exists only for sklearn estimator compatibility.

        Returns
        -------
        self : StandardScalerAR
            Fitted scaler instance.

        Notes
        -----
        The important distinction from sklearn.StandardScaler is that
        statistics are computed across *all values* in `x` rather than
        independently for each feature column.

        sklearn:
            mean_.shape == (n_features,)

        StandardScalerAR:
            mean_ is a single scalar value
        """
        # Compute global statistics across the entire matrix.
        # No axis is specified, so NumPy flattens the array before
        # calculating mean/variance/std.
        self.mean_ = np.mean(x)
        self.scale_ = np.std(x)
        self.var_ = np.var(x)

        # Store metadata expected by sklearn-like estimators.
        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, x):
        """
        Standardize data using the previously computed global statistics.

        Parameters
        ----------
        x : np.ndarray
            Data to scale.

        Returns
        -------
        np.ndarray
            Standardized data.

        Notes
        -----
        Transformation follows:

            z = (x - mean) / std

        where `mean` and `std` are scalar values computed from the
        entire training set.
        """
        x_tr = (x - self.mean_) / self.scale_
        return x_tr

    def inverse_transform(self, x):
        """
        Convert standardized data back to the original scale.

        Parameters
        ----------
        x : np.ndarray
            Standardized data.

        Returns
        -------
        np.ndarray
            Data restored to its original units.

        Notes
        -----
        Inverse transformation follows:

            x_original = z * std + mean
        """
        x_original = self.scale_ * x + self.mean_
        return x_original

    def fit_transform(self, x, y=None):
        """
        Fit the scaler and immediately transform the input data.

        Parameters
        ----------
        x : np.ndarray
            Training data.

        y : ignored, default=None
            Included for sklearn API compatibility.

        Returns
        -------
        np.ndarray
            Standardized version of `x`.
        """
        self.fit(x, y)
        return self.transform(x)

    def get_params(self):
        """
        Return estimator parameters.

        Included primarily for compatibility with sklearn pipelines,
        grid searches, and estimator utilities.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        return {}

    def set_params(self, **params):
        """
        Set estimator parameters.

        Included solely for sklearn compatibility.

        Parameters
        ----------
        **params
            Ignored parameters.

        Returns
        -------
        self : StandardScalerAR
            Estimator instance.
        """
        return self
