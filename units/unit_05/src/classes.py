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

    def get_params(self, deep=True):
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


class StandardScalerVAR:
    """
    Standard scaler designed specifically for vector autoregressive (VAR)
    and multivariable time series forecasting workflows.

    Unlike sklearn's StandardScaler, which computes separate statistics
    for every feature column, this scaler recognizes that the feature
    matrix generated for a VAR model consists of repeated lagged copies
    of the same process variables.

    For example, consider a VAR model containing:

        Biomass
        Glucose
        Product

    with:

        p = 3

    The lagged feature matrix may contain:

        [BIO(t-1), GLC(t-1), PRO(t-1), BIO(t-2), GLC(t-2), PRO(t-2), BIO(t-3), GLC(t-3), PRO(t-3)]

    Although these are nine columns, there are really only three underlying variables.

    This scaler computes a single mean and standard deviation for each process variable by
    combining all lagged versions of that variable together.

    Consequently:

        BIO(t-1), BIO(t-2), BIO(t-3)

    are all scaled using the same biomass statistics, while

        GLC(t-1), GLC(t-2), GLC(t-3)

    are all scaled using the same glucose statistics.

    This preserves consistency across lagged variables while still
    accounting for differences in scale between process variables.

    Attributes
    ----------
    p : int
        Number of lagged observations used by the VAR model.

    mean_ : np.ndarray
        Mean values used for scaling.

    scale_ : np.ndarray
        Standard deviations used for scaling.

    var_ : np.ndarray
        Variances used for scaling.

    n_samples_seen_ : int
        Number of observations used during fitting.

    n_features_in_ : int
        Number of columns in the input array.

    feature_names_in_ : array-like or None
        Placeholder attribute for sklearn API compatibility.
    """

    def __init__(self, p):
        """
        Initialize an unfitted VAR scaler.

        Parameters
        ----------
        p : int
            Number of lagged observations included in the VAR model.
        """

        self.p = p

        self.mean_ = None
        self.scale_ = None
        self.var_ = None

        self.n_samples_seen_ = None
        self.n_features_in_ = None

        self.feature_names_in_ = None

    def fit(self, x, y=None):
        """
        Compute scaling statistics for each process variable.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples, n_features * p)
            VAR lagged feature matrix.

        y : ignored, default=None
            Exists only for sklearn estimator compatibility.

        Returns
        -------
        self : StandardScalerVAR
            Fitted scaler instance.

        Notes
        -----
        Suppose the feature matrix contains:

            [BIO(t-1), GLC(t-1), BIO(t-2), GLC(t-2), BIO(t-3), GLC(t-3)]

        This method first separates the lagged states:

            Lag 1:
                [BIO(t-1), GLC(t-1)]

            Lag 2:
                [BIO(t-2), GLC(t-2)]

            Lag 3:
                [BIO(t-3), GLC(t-3)]

        These lagged blocks are then vertically stacked so that
        statistics can be computed for each process variable across
        all lagged observations.

        As a result:

            BIO(t-1), BIO(t-2), BIO(t-3)

        share a common biomass scaling distribution, while

            GLC(t-1), GLC(t-2), GLC(t-3)

        share a common glucose scaling distribution.
        """

        # Determine how many process variables are present.
        n_var_features = x.shape[1] // self.p

        # Ensure the number of columns is compatible with the specified autoregressive order.
        assert n_var_features * self.p == x.shape[1]

        # Separate each lag block:
        #
        #     Lag 1 --> columns corresponding to x(t-1)
        #     Lag 2 --> columns corresponding to x(t-2)
        #     ...
        #
        # Each block contains all process variables.
        x_sf = []
        for i in range(self.p):
            idx_start = i * n_var_features
            idx_end = idx_start + n_var_features
            x_sf.append(x[:, idx_start:idx_end])

        # Concatenate the lag blocks vertically (i.e., by rows).
        #
        # This creates a matrix where each column corresponds to a process variable and
        # all lagged observations contribute to the scaling statistics.
        x_sf = np.concatenate(x_sf, axis=0)

        # Calculate scaling statistics independently for each process variable.
        x_sf_mean = np.mean(x_sf, axis=0)
        x_sf_sd = np.std(x_sf, axis=0)
        x_sf_var = np.var(x_sf, axis=0)

        # Tile the statistics so that every lagged copy of a process variable uses identical scaling parameters.
        self.mean_ = np.tile(x_sf_mean, (1, self.p))
        self.scale_ = np.tile(x_sf_sd, (1, self.p))
        self.var_ = np.tile(x_sf_var, (1, self.p))

        # Store metadata expected by sklearn-like estimators.
        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, x):
        """
        Standardize data using the fitted scaling statistics.

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

        where mean and std are applied independently to each
        process variable.
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
            Standardized version of x.
        """
        self.fit(x, y)
        return self.transform(x)

    def get_params(self, deep=True):
        """
        Return estimator parameters.

        Included primarily for compatibility with sklearn pipelines,
        grid searches, and estimator utilities.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        return {"p": self.p}

    def set_params(self, **params):
        """
        Set estimator parameters.

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self : StandardScalerVAR
            Estimator instance.
        """

        if "p" in params:
            self.p = params["p"]

        return self
