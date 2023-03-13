import gc
import json
import os
import pdb

import h5py
import joblib
import numpy as np
import xgboost as xgb
from fire import Fire
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.callbacks import (CSVLogger, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimzers import Adam

from adcirc_rom.constants import SUPPORTED_MODELS
from adcirc_rom.model import (CorrelationFilter, FeatureImportanceFilter,
                              extract_features)


class StackedModel:
    """Similar to the one-shot XGBoost regression Model class

    The differences are that the sklearn api is used for XGBoost, and the problem
    is split into classification and regression steps.
    """

    def _get_modeldir(self, modelname):
        return f"{self._datadir}/models/{modelname}"

    def __init__(
        self,
        dataset="default",
        datadir="data",
        include_latlon=False,
        exclude_bathy=False,
        ordered_split=False,
    ):
        """Load in the dataset we will work with"""

        with h5py.File(f"{datadir}/datasets/{dataset}.hdf5", "r") as ds:
            print("Opened file")
            feature_values, self._feature_names = extract_features(
                ds, include_latlon=include_latlon, exclude_bathy=exclude_bathy
            )
            print("Loaded features")
            print(self._feature_names)
            maxele = ds["maxele"][:]
            maxele[maxele < 0] = 0
            self._storm_inds = ds["storm"][:]
            self._maxele = maxele
            self._coords = np.column_stack([ds["x"][:], ds["y"][:]])
            self._storm_names = ds["storm_names"][:]
            self.feature_values = feature_values

        self._datadir = datadir
        self._dataset = dataset
        self._ordered_split = ordered_split
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
        if self._ordered_split:
            unique_storms = np.unique(self._storm_inds)
            np.sort(unique_storms)
            nunique = len(unique_storms)
            pivot = unique_storms[int((1 - 1.0 / split_factor) * nunique)]
            print(f"Storms > {pivot} will be used for testing")
            train_mask = self._storm_inds <= pivot
            test_mask = ~train_mask
            self.holdout_inds = np.where(test_mask)[0]
            self.x_train, self.y_train = (
                self.feature_values[train_mask],
                self._maxele[train_mask],
            )
            self.x_test, self.y_test = (
                self.feature_values[test_mask],
                self._maxele[test_mask],
            )
            return

        dummy_arr = np.empty((len(self._storm_inds), 1))
        fold = GroupKFold(n_splits=split_factor)
        for train_inds, holdout_inds in fold.split(dummy_arr, groups=self._storm_inds):
            self.x_train = self.feature_values[train_inds, :]
            self.x_test = self.feature_values[holdout_inds, :]
            self.y_train = self._maxele[train_inds]
            self.y_test = self._maxele[holdout_inds]
            self.holdout_inds = holdout_inds
            break

    def train(
        self,
        classifier="nn1",
        regressor="nn1",
        epochs=100,
        preprocess=None,
        modelname=None,
        pca_components=50,
        correlation_threshold=0.9,
    ):
        """
        Trains the stacked model.

        Parameters
        ----------
        classifier : str, optional
            Classifier model name, by default "nn1".
        regressor : str, optional
            Regressor model name, by default "nn1".
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
        if classifier not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported classifier {classifier}!")
        if regressor not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported regressor {regressor}!")

        if modelname is None:
            modelname = f"stacked_{classifier}_{regressor}_{self._dataset}"
        modeldir = self._get_modeldir(modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir, exist_ok=True)

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

        y_train_class = np.zeros(len(y_train))
        y_train_class[y_train != 0] = 1
        x_train_normed = pipeline.fit_transform(x_train, y_train)
        x_test_normed = pipeline.transform(x_test)
        y_test_class = np.zeros(len(y_test))
        y_test_class[y_test != 0] = 1
        x_train_normed[~np.isfinite(x_train_normed)] = 0
        x_test_normed[~np.isfinite(x_test_normed)] = 0

        # save preprocesse data
        preproc_file = modeldir + "/preprocess_joblib"
        joblib.dump(pipeline, preproc_file)

        num_features = x_train_normed.shape[1]
        clf = self._get_model(classifier, num_features=num_features, classifier=True)
        reg = self._get_model(regressor, num_features=num_features, classifier=False)

        print("Training Classifier")
        if classifier.startswith("nn"):
            csv_logger = CSVLogger(
                modeldir + "/training.log", separator=",", append=False
            )
            cp = ModelCheckpoint(modeldir + "/classifier", save_best_only=True)
            callbacks = [
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,
                    patience=2,
                    min_lr=0.000001,
                    verbose=1,
                ),
                csv_logger,
                cp,
            ]
            optimizer = Adam(learning_rate=0.0001)
            clf.compile(
                loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            )
            clf.fit(
                x_train_normed,
                y_train_class,
                epochs=epochs // 2,
                batch_size=2048,
                validation_split=0.2,
                callbacks=callbacks,
            )
            test_stage1_pred = clf.predict(x_test_normed, batch_size=2048)
            test_stage1_pred = test_stage1_pred.flatten() > 0.5

        elif classifier.startswith("xgb"):
            # split the training data so we can do early stopping
            x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                x_train_normed, y_train_class, test_size=0.2
            )
            clf.fit(
                x_train_xgb,
                y_train_xgb,
                eval_set=[(x_val_xgb, y_val_xgb)],
                verbose=True,
            )
            os.makedirs(modeldir + "/classifier", exist_ok=True)
            clf.save_model(modeldir + "/classifier/model.xgb")
            test_stage1_pred = clf.predict(x_test_normed).astype(bool)

            print(confusion_matrix(y_test_class, clf.predict(x_test_normed)))
        elif classifier == "dummy":
            # don't perform classification
            test_stage1_pred = np.ones(len(x_test_normed)).astype(bool)

        acc = (test_stage1_pred.astype(int) == y_test_class).mean()
        print(f"Classification accuracy on test data {100*acc:.2f}%")

        pdb.set_trace()
        # train the regression model on non-zero values
        if classifier == "dummy":
            y_filter_index = np.ones(len(y_train)).astype(bool)
        else:
            y_filter_index = y_train != 0
        x_train_filter = x_train_normed[y_filter_index].copy()
        y_train_filter = y_train[y_filter_index].copy()
        gc.collect()

        print("Training regressor")
        if regressor.startswith("nn"):
            loss = MeanSquaredError(reduction="auto")
            optimizer = Adam(learning_rate=0.0001)
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
            )
            cp = ModelCheckpoint(modeldir + "/regressor", save_best_only=True)

            reg.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
            gc.collect()
            history = reg.fit(
                x_train_filter,
                y_train_filter,
                batch_size=2048,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[reduce_lr, cp],
            )
            gc.collect()

            # prediction pipline
            # prediction stage 2 is regression

            test_pred = np.zeros(x_test.shape[0])
            test_pred[test_stage1_pred] = reg.predict(
                x_test_normed[test_stage1_pred, :], batch_size=2048
            ).reshape(-1)

            gc.collect()
        else:
            # split the training data so we can do early stopping
            x_train_xgb, x_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                x_train_filter, y_train_filter, test_size=0.2
            )
            reg.fit(
                x_train_xgb,
                y_train_xgb,
                eval_set=[(x_val_xgb, y_val_xgb)],
                verbose=True,
            )
            os.makedirs(modeldir + "/regressor", exist_ok=True)
            reg.save_model(modeldir + "/regressor/model.xgb")
            test_pred = np.zeros(x_test.shape[0])
            test_pred[test_stage1_pred] = reg.predict(
                x_test_normed[test_stage1_pred, :]
            )

        # Absolute error on predictions
        error_test = np.abs(y_test.flatten() - test_pred.flatten())
        mae = error_test.mean()
        rmse = (error_test**2).mean() ** 0.5
        res = {"accuracy": acc, "mae": mae, "rmse": rmse}
        print(res)
        with open(modeldir + "/results.json", "w") as fp:
            json.dump(res, fp)

        # Save the predictions for later plotting
        with h5py.File(modeldir + "/test_preds.hdf5", "w") as outds:
            outds["test_pred"] = test_pred
            outds["storm_inds"] = self._storm_inds[self.holdout_inds]
            outds["coords"] = self._coords[self.holdout_inds]
            outds["maxele"] = y_test

        return res

    def _get_model(self, name, num_features, classifier=True):
        params = SUPPORTED_MODELS[name]
        params["classifier"] = classifier
        if name.startswith("nn"):
            return self._get_nn(num_features=num_features, **params)
        elif name.startswith("xgb"):
            return self._get_xgb(**params)

    def _get_nn(self, num_features, size=1, classifier=True):
        inputs = keras.Input(shape=num_features)
        initial_width = width = 256
        x = keras.layers.Dense(initial_width, activation="relu")(inputs)

        for i in range(size):
            width *= 2
            x = keras.layers.Dense(width, activation="relu")(x)

        for i in range(size):
            width = width // 2
            x = keras.layers.Dense(width, activation="relu")(x)

        if classifier:
            x = keras.layers.Dense(1, activation="sigmoid")(x)
        else:
            x = keras.layers.Dense(1, activation="relu")(x)

        return keras.Model(inputs, x)

    def _get_xgb(self, classifier=True, **kwargs):
        if classifier:
            return xgb.XGBClassifier(eval_metric="error", **kwargs)
        else:
            return xgb.XGBRegressor(eval_metric="mae", **kwargs)

    def predict(self, modelname, test_only=False):
        """
        Generate predictions for the given dataset

        If test_only is True, assume this is the original dataset the model was
        trained with and regenerate predictions for the test set. Otherwise,
        generated named predictions for the entire datset.

        Parameters
        ----------
        modelname : str
            Name of the trained model to use for prediction.
        test_only : bool, optional
            Whether to generate predictions for the test set only, by default
        False.

        Returns
        -------
        outname: str
            Name of output file named 'test_pred5.hdf5' if test_only is set,
            or '{self._dataset}._preds.hdf5' if it is not.
        """

        # load model
        modeldir = self._get_modeldir(modelname)
        # for now we just support neural nets
        pipeline = joblib.load(modeldir + "/preprocess_joblib")
        if test_only:
            self._split_data()
            X, y = self.x_test, self.y_test
            coords = self._coords[self.holdout_inds]
            storm_inds = self._storm_inds[self.holdout_inds]
        else:
            X, y = self.feature_values, self._maxele
            coords = self._coords
            storm_inds = self._storm_inds

        X = pipeline.transform(X)

        if os.path.exists(modeldir + "/classifier/model.xgb"):
            classifier = xgb.XGBClassifier()
            classifier.load_model(modeldir + "/classifier/model.xgb")
            inundation_flag = classifier.predict(X).astype(bool)
        elif "dummy" in modeldir:
            inundation_flag = np.ones(X.shape[0]).astype(bool)
        else:
            classifier = keras.models.load_model(modeldir + "/classifier")
            inundation_flag = (
                classifier.predict(X, batch_size=2048).reshape(-1) > 0.5
            ).astype(bool)

        acc = (inundation_flag == (y != 0)).mean()
        print(f"Classification accuracy {100*acc:2f}")
        elevations = np.zeros(X.shape[0])

        if os.path.exists(modeldir + "/regressor/model.xgb"):
            regressor = xgb.XGBRegressor()
            regressor.load_model(modeldir + "/regressor/model.xgb")
            elevations[inundation_flag] = regressor.predict(X[inundation_flag])
        else:
            regressor = keras.models.load_model(modeldir + "/regressor")
            elevations[inundation_flag] = regressor.predict(
                X[inundation_flag], batch_size=2048
            ).reshape(-1)

        mae = np.abs((elevations - y)).mean()
        rmse = ((elevations - y) ** 2).mean() ** 0.5
        print(f"mae: {mae}, rmse: {rmse}")

        outname = "test_preds.hdf5" if test_only else f"{self._dataset}_preds.hdf5"
        with h5py.File(modeldir + "/" + outname, "w") as outds:
            if test_only:
                outds["test_pred"] = elevations
            else:
                outds["pred"] = elevations

            outds["coords"] = coords
            outds["storm_inds"] = storm_inds
            outds["maxele"] = y

        return outname


if __name__ == "__main__":
    Fire(StackedModel)
