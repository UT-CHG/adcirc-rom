from fire import Fire
import xgboost as xgb
import h5py
import numpy as np
from sklearn.model_selection import GroupKFold
from collections import defaultdict
import os
import pandas as pd

class FeatureImportance(xgb.callback.TrainingCallback):
    """Monitors the feature importances during cross-validation. 
    """

    def __init__(self, sample_rounds=50):
        """sample_rouds defines how often we sample the feature importances
        """

        self.feature_importances = defaultdict(list)
        self.sample_rounds = sample_rounds


    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.sample_rounds: return
        bst = model.cvfolds[0].bst
        importances = bst.get_score(importance_type='gain')
        for k, v in importances.items(): self.feature_importances[k].append(v)

    def print_summary(self, top=10):
        feat_names = []
        avg_importances = []
        for k, v in self.feature_importances.items():
            feat_names.append(k)
            avg_importances.append(np.mean(np.array(v)))
        avg_importances = np.array(avg_importances)
        order = np.argsort(avg_importances)
        print(f"Top {top} Features")
        for i, ind in enumerate(order[-1:-top-1:-1]):
            print(f"#{i+1}: {feat_names[ind]} with gain {avg_importances[ind]:.2f}")

def extract_features(ds):
    arrs = []
    names = []
    # add forcing
    for var in ["windx", "windy", "pressure"]:
        arr = ds[var][:]
        arrs.extend([np.max(arr, axis=1), np.mean(arr, axis=1), np.min(arr, axis=1)])
        names.extend([f"{var}_{func}" for func in ["max", "mean", "min"]])
        #arrs.append(arr)
        #names.extend([f"{var}_{i}" for i in range(arr.shape[1])])

    # add scalar properties
    for var in ["coastal_dist", "landfall_dist", "depth"]:
        # use np.newaxis to make array 2-d so we can call np.concatenate
        arrs.append(ds[var][:][:, np.newaxis])
        names.append(var)

    return np.column_stack(arrs), names

class XGBModel:
    """A class to handle loading and saving XGB model files.
    """
    
    def __init__(self, dataset="default", datadir="data"):
        """Load in the dataset we will work with
        """

        with h5py.File(f"{datadir}/datasets/{dataset}.hdf5", "r") as ds:
            feature_values, self._feature_names = extract_features(ds)
            maxele = ds["maxele"][:]
            maxele[maxele<0] = 0
            self._storm_inds = ds["storm"][:]
            self._maxele = maxele

        self.dmat = xgb.DMatrix(feature_values, label=maxele, feature_names=self._feature_names)
        self._datadir = datadir
        self._dataset = dataset

    def _split_data(self, seed=2022, split_factor=10):
        # reproducibility
        np.random.seed(2022)
        dummy_arr = np.empty((len(self._storm_inds), 1))
        fold = GroupKFold(n_splits=split_factor)
        for train_inds, holdout_inds in fold.split(dummy_arr, groups=self._storm_inds):
            self.dtrain = self.dmat.slice(train_inds)
            self.dholdout = self.dmat.slice(holdout_inds)
            self.train_inds = train_inds
            self.holdout_inds = holdout_inds
            break

    def _get_modeldir(self, name):
        return f"{self._datadir}/models/{name}"
            
    def train(self, name, num_boost_round=200, **kwargs):
        """Train the model 
        """

        self._split_data()
        params = {**kwargs}
        bst = xgb.train(params, self.dtrain, num_boost_round=num_boost_round, verbose_eval=10)
        preds = bst.predict(self.dholdout)
        output_modeldir = self._get_modeldir(name)

        if not os.path.exists(output_modeldir):
            os.makedirs(output_modeldir)
        bst.save_model(f"{output_modeldir}/model.xgb")
        with h5py.File(f"{output_modeldir}/preds.hdf5", "w") as outf:
            outf["pred"] = preds
            outf["storm"] = self._storm_inds[self.holdout_inds],
            outf["feature_names"] = np.array(self._feature_names, dtype="S")
            outf["maxele"] = self._maxele[self.holdout_inds]
            params["num_boost_round"] = num_boost_round
            params["dataset"] = self._dataset
            for p, v in params.items():
                outf.attrs[p] = v

        
    def cv(self, num_boost_round=200, folds=5, **kwargs):
        """Run cross-validation
        """

        self._split_data()

        importances = FeatureImportance()
        folds = list(GroupKFold(n_splits=5).split(np.empty((self.dtrain.num_row(),1)), groups=self._storm_inds[self.train_inds]))

        ev_hist = xgb.cv(kwargs, self.dtrain, num_boost_round=num_boost_round, folds=folds,
                        verbose_eval=10, callbacks=[importances])
        importances.print_summary(top=20)
        print(ev_hist)
                                                                                
    def predict(self, name):
        """Predict with the given dataset and save the results to the model directory
        """

        bst = xgb.Booster()
        bst.load_model(self._get_modeldir(name)+"/model.xgb")
        preds = bst.predict(self.dmat)
        preds[preds<0] = 0
        errs = preds - self._maxele
        
        mae = np.mean(np.abs(errs))
        rmse = (errs**2).mean() ** .5
        median = np.median(np.abs(errs))
        print(f"mae={mae}, rmse={rmse}, median={median}, bias={errs.mean()}")
        print(pd.DataFrame({"pred": preds, "maxele": self._maxele}).describe())
        
if __name__ == "__main__":
    Fire(XGBModel)