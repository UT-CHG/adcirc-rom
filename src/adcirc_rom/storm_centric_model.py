import gc
#import json
#import os
import pandas as pd

#import h5py
#import joblib
import numpy as np
import xgboost as xgb
from fire import Fire
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from adcirc_rom.features import (CorrelationFilter, FeatureImportanceFilter)

class StormCentricModel:
    """Simple class for an XGBoost regression problem"""

    def __init__(
        self,
        dataset="storm_centric_dataset.csv",
        basin=None,
        clip=False,
        category=None
    ):
        """Load in the dataset we will work with"""

        df = pd.read_csv(dataset)
        target = "max_zeta_diff"
        bad_cols = ["lat", "lon", "basin", target, "category"]
        features = [c for c in df.columns if not any(b in c for b in bad_cols)]
        if clip:
          df = df[(df[target] < 6) & (df[target]>1)]
        if basin is not None:
            if basin == "NA":
                df = df[df["basin"].isna()]
            else:
                df = df[df["basin"] == basin]
        if category is not None: df = df[df["category"] == category]
        self._maxele = df[target].values
        self.feature_values = df[features].values
        print(df.groupby("basin", dropna=False)[target].describe())
        print("Loaded data")

    def _split_data(self, split_factor=10, seed=2022):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        split_factor : int, optional
            Number of splits for cross-validation, by default 10.
        seed : int, optional
            Seed for random number generator, by default 2022.

        Returns
        -------
        None
            The function sets the following attributes: `x_train`, `y_train`,
            `x_test`,  `y_test`, and `holdout_inds`.

        Note
        ----
        If function has been called before, split is not re-computed.
        """
        if hasattr(self, "x_train"):
            return
        np.random.seed(seed)
        fold = KFold(n_splits=split_factor)
        for train_inds, holdout_inds in fold.split(self.feature_values):
            self.x_train = self.feature_values[train_inds, :]
            self.x_test = self.feature_values[holdout_inds, :]
            self.y_train = self._maxele[train_inds]
            self.y_test = self._maxele[holdout_inds]
            self.holdout_inds = holdout_inds
            break

    def train(
        self,
        preprocess=None,
        modelname=None,
        pca_components=50,
        correlation_threshold=0.9,
    ):
        """
        Trains the model.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs, by default 100.
        preprocess : str or None, optional
            Preprocessing method, by default None.
        modelname : str or None, optional
            Name for the trained model, by default will resolve to:
            `stacked_{classifier}_{regressor}_{self._dataset}`
        pca_components : int, optional
            Number of PCA components to keep, by default 50.
        correlation_threshold : float, optional
            Correlation threshold for feature selection, by default 0.9.

        Returns
        -------
        res : dict
            Dictionary with results of training the model, including the
            classification accuracy, the mean error in regression, and the
            root mean squared error.
        """

        self._split_data()

        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test

        transforms = []
        if preprocess == "pca":
            transforms.append(("pca", PCA(n_components=pca_components)))
        elif preprocess == "importance":
            transforms.append(("feature_importance", FeatureImportanceFilter()))
        elif preprocess == "correlation":
            transforms.append(
                ("corr", CorrelationFilter(threshold=correlation_threshold))
            )
        elif preprocess is not None:
            raise ValueError(f"Unrecognized preprocess scheme {preprocess}")

        transforms.append(("scaler", StandardScaler()))
        pipeline = Pipeline(transforms)

        x_train_normed = pipeline.fit_transform(x_train, y_train)
        x_test_normed = pipeline.transform(x_test)
        x_train_normed[~np.isfinite(x_train_normed)] = 0
        x_test_normed[~np.isfinite(x_test_normed)] = 0
        print(x_train_normed.shape, x_train.shape)
        # save preprocesse data
        #preproc_file = modeldir + "/preprocess_joblib"
        #joblib.dump(pipeline, preproc_file)

        num_features = x_train_normed.shape[1]
        reg = xgb.XGBRegressor(eval_metric="mae")

        # split the training data so we can do early stopping
        x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                x_train_normed, y_train, test_size=0.2
         )
        reg.fit(
            x_train_xgb,
            y_train_xgb,
            eval_set=[(x_val_xgb, y_val_xgb)],
            verbose=True,
        )
        #os.makedirs(modeldir + "/regressor", exist_ok=True)
        #reg.save_model(modeldir + "/regressor/model.xgb")
        test_pred = reg.predict(
            x_test_normed
        )
        df = pd.DataFrame({"pred": test_pred, "true": y_test})
        print(df.describe())
        #test_pred[:] = test_pred.mean()
        # Absolute error on predictions
        error_test = np.abs(y_test.flatten() - test_pred.flatten())
        mae = error_test.mean()
        rmse = (error_test**2).mean() ** 0.5
        res = {"mae": mae, "rmse": rmse}
        print("R^2", r2_score(y_test, test_pred))
        print(res)
        #with open(modeldir + "/results.json", "w") as fp:
        #    json.dump(res, fp)

        # Save the predictions for later plotting
        #with h5py.File(modeldir + "/test_preds.hdf5", "w") as outds:
        #    outds["test_pred"] = test_pred
        #    outds["storm_inds"] = self._storm_inds[self.holdout_inds]
        #    outds["coords"] = self._coords[self.holdout_inds]
        #    outds["maxele"] = y_test

        return res

if __name__ == "__main__":
    Fire(StormCentricModel)
