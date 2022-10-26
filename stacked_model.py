import h5py
import os
from model import extract_features
import gc
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, ModelCheckpoint
from fire import Fire
import numpy as np

class StackedModel:
    """Similar to the one-shot XGBoost regression Model class

    The differences are that the sklearn api is used for XGBoost, and the problem
    is split into classification and regression steps.
    """

    supported_models = {
        "nn1": {
            "size":1
        },
        "nn2": {
            "size":2
        },
        "nn3": {
            "size":3
        },
        "xgb250": {
            "n_estimators": 250
        }
    }
    
    
    def __init__(self, dataset="default", datadir="data",
                 include_latlon=False, exclude_bathy=False,
                 split_factor=10, seed=2022):
        """Load in the dataset we will work with
        """

        with h5py.File(f"{datadir}/datasets/{dataset}.hdf5", "r") as ds:
            print("Opened file")
            feature_values, self._feature_names = extract_features(
                ds, include_latlon=include_latlon, exclude_bathy=exclude_bathy
            )
            print("Loaded features")
            maxele = ds["maxele"][:]
            maxele[maxele<0] = 0
            self._storm_inds = ds["storm"][:]
            self._maxele = maxele
            self._coords = np.column_stack([ds["x"][:], ds["y"][:]])
            self._storm_names = ds["storm_names"][:]

        self._datadir = datadir
        self._dataset = dataset

        np.random.seed(seed)
        dummy_arr = np.empty((len(self._storm_inds), 1))
        fold = GroupKFold(n_splits=split_factor)
        for train_inds, holdout_inds in fold.split(dummy_arr, groups=self._storm_inds):
            self.x_train = feature_values[train_inds,:]
            self.x_test = feature_values[holdout_inds,:]
            self.y_train = maxele[train_inds]
            self.y_test = maxele[holdout_inds]
            break

        del feature_values
        gc.collect()
        print("Loaded and split data.")

    
    def train(self, classifier="nn1", regressor="nn1",
             epochs=100):
        """Train the stacked model
        """
        
        modelname = f"stacked_{classifier}_{regressor}_{self._dataset}"
        modeldir = f"{self._datadir}/models/{modelname}"
        if not os.path.exists(modeldir): os.makedirs(modeldir, exist_ok=True)
        
        if classifier not in self.supported_models:
            raise ValueError(f"Unsupported classifier {classifier}!")
        if regressor not in self.supported_models:
            raise ValueError(f"Unsupported regressor {regressor}!")
    
        clf = self._get_model(classifier, classifier=True)      
        reg = self._get_model(regressor, classifier=False)
        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test
    
        y_train_class = np.zeros(len(y_train))
        y_train_class[y_train!=0] = 1
        xmeans = np.mean(x_train, axis=0)
        # print("xmeans", xmeans)
        xstds = np.std(x_train, axis=0, ddof=1)
        # print("xstds", xstds)

        x_train_normed = (x_train - xmeans) / xstds
        x_test_normed = (x_test - xmeans) / xstds
        y_test_class = np.zeros(len(y_test))
        y_test_class[y_test!=0] = 1
        
        print("Training Classifier")
        if classifier.startswith("nn"):
            csv_logger = CSVLogger(modeldir+'/training.log', separator=',', append=False)
            cp = ModelCheckpoint(modeldir+'/classifier', save_best_only=True)
            callbacks = [
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=2, min_lr=0.000001, verbose=1),
                    csv_logger,
                    cp
                ]
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            clf.compile(loss='binary_crossentropy',
                optimizer= optimizer,
                metrics=['accuracy'])
            clf.fit(x_train_normed, y_train_class, epochs=epochs//2,
                    batch_size=2048, validation_split = 0.2,
                    callbacks = callbacks)
        else:
            print("Accuracy on Train Data: {:.2f}%".format(clf.score(x_train,y_train_class)*100))
            print("Accuracy on Test Data: {:.2f}%".format(clf.score(x_test,y_test_class)*100))
            print(confusion_matrix(y_test_class, clf.predict(x_test)))
        #train the regression model on non-zero values
        y_filter_index = y_train!=0
        x_train_filter = x_train_normed[y_filter_index].copy()
        y_train_filter = y_train[y_filter_index].copy()
        gc.collect()

        print("Training regressor")
        if regressor.startswith("nn"):
            #prediction pipline
            #prediction stage 1 is classification
            test_stage1_pred = clf.predict(x_test_normed, batch_size=2048)
            test_stage1_pred = (test_stage1_pred.flatten() > 0.5)
            print(test_stage1_pred)
            acc = (test_stage1_pred.astype(int)==y_test_class).mean()
            print(f"Classification accuracy on test data {acc:.4f}")
            gc.collect()

            loss = tf.keras.losses.MeanSquaredError(reduction="auto")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, min_lr=0.00001)
            reg.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
            gc.collect()
            history = reg.fit(x_train_filter, y_train_filter,
                                batch_size = 2048, epochs=epochs,
                                validation_split = 0.2, callbacks = [reduce_lr])
            gc.collect()

            #prediction pipline
            #prediction stage 2 is regression

            test_pred = np.zeros(x_test.shape[0])
            test_pred[test_stage1_pred] = reg.predict(x_test_normed[test_stage1_pred,:], batch_size=2048).reshape(-1)

            gc.collect()
            #Absolute error on predictions
            error_test = np.abs(y_test.reshape(-1,1) - test_pred.reshape(-1,1))
            print("Absolute Error on Test Data : {:.2f} m".format(error_test.mean()))
        
    
    def _get_model(self, name, classifier=True):
        params = self.supported_models[name]
        params["classifier"] = classifier
        if name.startswith("nn"):
            return self._get_nn(**params)
        elif name.startswith("xgb"):
            return self._get_xgb(**params)
    
    def _get_nn(self, size=1, classifier=True):
        inputs = keras.Input(shape=self.x_train.shape[1])
        initial_width = width = 256
        x = layers.Dense(initial_width, activation="relu")(inputs)

        for i in range(size):
            width *= 2
            x = layers.Dense(width, activation="relu")(x)

        for i in range(size):
            width = width // 2
            x = layers.Dense(width, activation="relu")(x)
        
        if classifier:
            x = layers.Dense(1, activation="sigmoid")(x)
        else:
            x = layers.Dense(1, activation="relu")(x)
        
        return keras.Model(inputs, x)

if __name__ == "__main__":
    Fire(StackedModel)