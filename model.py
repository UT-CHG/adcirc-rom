from fire import Fire
import xgboost as xgb
import h5py
import numpy as np
from sklearn.model_selection import GroupKFold
from collections import defaultdict
import os
import pandas as pd
import gc

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
        self.add_importances(bst)

    def add_importances(self, bst):    
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

def extract_features(ds, include_latlon=False, exclude_bathy=False):
    arrs = []
    names = []

    # add scalar properties
    for var in sorted(list(ds.keys())):
        #if var.count("_") >= 3: continue
        if ((not exclude_bathy and var.startswith("bathy")) 
            or var in ["coastal_dist", "landfall_dist", "depth"]
            or (include_latlon and var in ["x", "y"])
            or any([k in var for k in ["wind", "pressure", "magnitude"]])
           ):
            names.append(var)

    n = ds[names[0]].shape[0]
    mat = np.zeros((len(names), n))
    for i, name in enumerate(names):
        mat[i] = ds[name][:]

    return mat.T, names            
            
class XGBModel:
    """A class to handle loading and saving XGB model files.
    """
    
    def __init__(self, dataset="default", datadir="data", include_latlon=False, exclude_bathy=False):
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

        self.dmat = xgb.DMatrix(feature_values, label=maxele,
                                nthread=32,
                                feature_names=self._feature_names)
        self._datadir = datadir
        self._dataset = dataset
        del feature_values
        gc.collect()
        print("Loaded DMatrix")

            
    def _split_data(self, seed=2022, split_factor=10):
        # reproducibility
        np.random.seed(seed)
        dummy_arr = np.empty((len(self._storm_inds), 1))
        fold = GroupKFold(n_splits=split_factor)
        for train_inds, holdout_inds in fold.split(dummy_arr, groups=self._storm_inds):
            self.dtrain = self.dmat.slice(train_inds)
            self.dholdout = self.dmat.slice(holdout_inds)
            self.train_inds = train_inds
            self.holdout_inds = holdout_inds
            break
        # after slicing we don't need the original matrix
        del self.dmat
        gc.collect()
        print("Split DMatrix")
            
    def _get_modeldir(self, name):
        return f"{self._datadir}/models/{name}"

    def _fix_nthread(self, kwargs):
        if "nthread" not in kwargs: kwargs["nthread"] = 64
    
    def train(self, name, num_boost_round=200, **kwargs):
        """Train the model 
        """

        self._split_data()
        self._fix_nthread(kwargs)

        params = {**kwargs}
        bst = xgb.train(params, self.dtrain, num_boost_round=num_boost_round,
                        evals=[(self.dtrain, "train")], verbose_eval=10)
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

        
    def cv(self, num_boost_round=20, folds=3, custom=False, **kwargs):
        """Run cross-validation
        """

        self._split_data()
        self._fix_nthread(kwargs)

        folds = list(GroupKFold(n_splits=folds).split(np.empty((self.dtrain.num_row(),1)), groups=self._storm_inds[self.train_inds]))

        print("Starting CV")
        if custom:
            ev_hist = self._custom_cv(kwargs,
                                      num_boost_round=num_boost_round, folds=folds)
        else:
            importances = FeatureImportance()
            ev_hist = xgb.cv(kwargs, self.dtrain, num_boost_round=num_boost_round, folds=folds,
                        verbose_eval=5, callbacks=[importances])
            importances.print_summary(top=20)
        print(ev_hist)

    def _custom_cv(self, params, num_boost_round, folds):
        """A more efficient cross-validation function
        """

        for train_inds, val_inds in folds:
            train_mat = self.dtrain.slice(train_inds)
            val_mat = self.dtrain.slice(val_inds)
            bst = xgb.train(params, train_mat, num_boost_round=num_boost_round,
                            evals=[(train_mat, "train"),(val_mat, "val")], verbose_eval=10)
            f = FeatureImportance()
            f.add_importances(bst)
            f.print_summary(top=20)
            del train_mat, val_mat
            
        
    def predict(self, name):
        """Predict with the given dataset and save the results to the model directory
        """

        bst = xgb.Booster()
        modeldir = self._get_modeldir(name)
        bst.load_model(modeldir+"/model.xgb")
        preds = bst.predict(self.dmat)
        preds[preds<0] = 0
        errs = preds - self._maxele
        
        mae = np.mean(np.abs(errs))
        rmse = (errs**2).mean() ** .5
        median = np.median(np.abs(errs))
        print(f"mae={mae}, rmse={rmse}, median={median}, bias={errs.mean()}")
        print(pd.DataFrame({"pred": preds, "maxele": self._maxele}).describe())
        with h5py.File(f"{modeldir}/{self._dataset}_preds.hdf5", "w") as outf:
            outf["pred"] = preds
            outf["maxele"] = self._maxele
            outf["storm"] = self._storm_inds
            outf["storm_names"] = self._storm_names
            outf["coords"] = self._coords
        
if __name__ == "__main__":
    Fire(XGBModel)