import dcor
from glob import glob
from joblib import Parallel, delayed
import json
import numpy as np
import os
from os.path import join
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class DeepRepVizBackend:
    """Backend for DeepRepViz."""

    def __init__(
        self,
        conf_table=None,
        ID_col="ID",
        label_col="label",
        best_ckpt_by="loss_train",
        best_ckpt_metric_should_be="min",
        debug=False,
    ):
        self.debug = debug
        # check and prepare the conf table and prepare it
        if isinstance(conf_table, str) and os.path.isfile(conf_table):
            conf_table = pd.read_csv(conf_table)
        elif isinstance(conf_table, pd.DataFrame):
            conf_table = conf_table.copy()
        else:
            # raise a warning if the conf_table is not provided
            warnings.warn(
                f"Provided conf_table '{conf_table}' is neither a pandas Dataframe nor a path to a table.\
 Skipping all conf_table operations.. "
            )
        if conf_table is not None:
            self.label_col = label_col
            self.ID_col = ID_col
            assert self.label_col in conf_table.columns
            assert (
                self.ID_col in conf_table.columns and conf_table[self.ID_col].is_unique
            ), f"the conf_table MUST have an ID \
    column with provided name '{self.ID_col}' that has a unique value for each datapoint."
            self.conf_table = conf_table.set_index(
                self.ID_col
            )  # .astype({self.ID_col: str})

            self.df_conf = self._preprocess_conf_table(
                self.conf_table, ID_col, label_col
            )

        self.deeprepvizlogs = {}
        self.best_ckpt_by = best_ckpt_by
        self.best_ckpt_metric_should_be = best_ckpt_metric_should_be

    def _preprocess_conf_table(self, conf_table, ID_col="ID", label_col="label"):
        """Clean and process the confounds table such as preprocess the
        covariates to numerical values that can be used for metric computations"""
        df_confs = pd.DataFrame(columns=conf_table.columns)
        for c in conf_table.columns:
            df_confs[c] = [np.nan] * len(conf_table)
        conf_dtypes = {}

        categorical_dtypes = ["object", "bool", "category"]
        if self.debug:
            print("[log] Potential Confounders to be assessed:")
        for conf_col in df_confs.columns:
            c = conf_table[conf_col]

            # if a conf column is categorical
            if (c.dtype.name in categorical_dtypes) or c.nunique() == 2:
                # now encode the values with sklearn's labelencoder()
                encoder = preprocessing.LabelEncoder()
                mask_non_nan = ~pd.isnull(c.values)
                c_vals = c.values[mask_non_nan]
                df_confs.loc[mask_non_nan, conf_col] = encoder.fit_transform(c_vals)
                if c.nunique() == 2:
                    pred_type = "classif_binary"
                else:
                    pred_type = "classif"
                    df_confs[conf_col] = df_confs[conf_col].astype(int)
            else:
                # scale continuous values using min max scalar
                encoder = preprocessing.MinMaxScaler()
                mask_non_nan = ~pd.isnull(c.values)
                c_vals = c.values[mask_non_nan].reshape(
                    -1, 1
                )  # minmax expects 2D input
                df_confs.loc[mask_non_nan, conf_col] = encoder.fit_transform(c_vals)
                pred_type = "regression"
                df_confs[conf_col] = df_confs[conf_col].astype(float)

            appendlog = (
                f"uniques = {dict(c.value_counts(dropna=False))}"
                if pred_type != "regression"
                else ""
            )
            if self.debug:
                print(
                    f"[log] \t {c.name} {' '*(25-len(c.name))} predtype: {pred_type}  {' '*(15-len(pred_type))} {appendlog}"
                )
            conf_dtypes.update({c.name: pred_type})

        self.conf_dtypes = conf_dtypes

        return df_confs

    def load_log(self, logsdir):
        """Load all DeepRepViz log/checkpoints present in the directory"""
        deeprepvizlog = {}
        assert os.path.isdir(logsdir)

        best_ckpt_idx = -1
        best_ckpt_score = np.inf if self.best_ckpt_metric_should_be == "min" else 0
        checkpoints = []
        for i, ckpt_dir in enumerate(sorted(glob(logsdir + "/*/*/"))):
            ckpt_name = os.path.basename(os.path.normpath(ckpt_dir))
            # (1) load the tensors.tsv
            ckpt_values = {
                "acts": pd.read_csv(
                    ckpt_dir + "tensors.tsv", sep="\t", header=None
                ).values
            }
            # if the tensors_3D.tsv is present then load it too
            if os.path.isfile(ckpt_dir + "tensors_3D.tsv"):
                ckpt_values["acts_3D"] = pd.read_csv(
                    ckpt_dir + "tensors_3D.tsv", sep="\t", header=None
                ).values
            # (2) read the metadata.tsv columns
            metadata = pd.read_csv(ckpt_dir + "metadata.tsv", sep="\t")
            if "IDs" not in deeprepvizlog:
                # TODO implement a cross check that the IDs in the self.conf_table are present in the log IDs and the labels also match
                deeprepvizlog["IDs"] = metadata["IDs"].values
                deeprepvizlog["labels"] = metadata["labels"].values
            ckpt_values.update(
                {
                    col: metadata[col].values
                    for col in metadata
                    if col not in ["IDs", "labels"]
                }
            )

            # (3) load metrics
            with open(ckpt_dir + "metametadata.json") as fp:
                metametadata = json.load(fp)
                ckpt_values.update(metametadata)
                score = metametadata["metrics"][self.best_ckpt_by]
                if (
                    self.best_ckpt_metric_should_be == "min" and score < best_ckpt_score
                ) or (
                    self.best_ckpt_metric_should_be == "max" and score > best_ckpt_score
                ):
                    best_ckpt_score = score
                    best_ckpt_idx = i

            checkpoints.append((ckpt_name, ckpt_values))

        deeprepvizlog["best_ckpt_idx"] = best_ckpt_idx
        deeprepvizlog["checkpoints"] = checkpoints
        self.deeprepvizlogs.update({logsdir: deeprepvizlog})

        # save the computed checkpoint information in the deeprepvizlog folder as a json file

        with open(join(logsdir, 'best_checkpoint.json'), 'w') as fp:
            if 'train' in self.best_ckpt_by:
                # throw a warning message that the best checkpoint is not decided on a test metric
                warnings.warn(f"The 'best_checkpoint' is decided based on the '{self.best_ckpt_by}' metric \
which is calculated on the training data. It is recommended to configure DeepRepVizBackend(best_ckpt_by=..) \
to a test or validation data metric instead.")
            json.dump({
                'ckpt_idx' : best_ckpt_idx,
                'ckpt_name': checkpoints[best_ckpt_idx][0],
                'metrics'  : checkpoints[best_ckpt_idx][1]['metrics']
            }, fp, indent=4)

        with open(join(logsdir, "best_checkpoint.json"), "w") as fp:
            json.dump(
                {
                    "ckpt_idx": best_ckpt_idx,
                    "ckpt_name": checkpoints[best_ckpt_idx][0],
                    "metrics": checkpoints[best_ckpt_idx][1]["metrics"],
                },
                fp,
                indent=4,
            )

    def __str__(self):
        self._pprint_deeprepvizlogs()

    def _pprint_deeprepvizlogs(self):
        if self.deeprepvizlogs == {}:
            print(
                "No DeepRepViz logs loaded yet. Please load the logs using the load_log() method first."
            )
        else:
            for i, (log_key, log_dict) in enumerate(self.deeprepvizlogs.items(), 1):
                print(f"{'-'*100}\nDeepRepViz log no.{i}: \n{log_key}")
                print(f"\tn(samples)        : {len(log_dict['IDs'])}")
                print(
                    f"\tIDs               : {log_dict['IDs'][:3]} ... {log_dict['IDs'][-3:]}"
                )
                print(
                    f"\tlabels            : {log_dict['labels'][:3]} ... {log_dict['labels'][-3:]}"
                )
                print(f"\tn(checkpoints)    : {len(log_dict['checkpoints'])}")
                best_ckpt = log_dict["checkpoints"][log_dict["best_ckpt_idx"]]
                # print the other keys present in the log_dict
                print(f"\t\t\t   Each checkpoint has the following data: ")
                for k, v in best_ckpt[1].items():
                    if isinstance(v, np.ndarray):
                        print(f"\t\t\t   {k}.shape : {v.shape}")
                    else:
                        show_val = ""
                        if isinstance(v, dict):
                            show_val = f"| keys: {list(v.keys())}"
                        elif isinstance(v, list):
                            show_val = f"| len: {len(v)}"
                        print(f"\t\t\t   {k} : {type(v)} {show_val}")
                print(
                    f"\tbest checkpoint   : best_ckpt_idx = {log_dict['best_ckpt_idx']}) \t name = {best_ckpt[0]}"
                )
                print(
                    "\t\t\t   {}".format(
                        {
                            k: "{:.2f}".format(v)
                            for k, v in best_ckpt[1]["metrics"].items()
                        }
                    )
                )

                print(f"\n\tattributes table  : shape = {self.conf_table.shape}")
                attr_cols = [
                    c
                    for c in self.conf_table.columns
                    if c not in [self.ID_col, self.label_col]
                ]
                print(
                    f"\t\t\t   ID_col = {self.ID_col} \t label_col = {self.label_col}  \t n(attrs cols) = {len(attr_cols)}"
                )
                print(f"\t\t\t   attrs = {attr_cols[:3]} ... {attr_cols[-3:]}")
                print(f"{'-'*100}")

    ###############################         downsampling        ########################################

    def _is_downsampled(self, log_key):
        """Check if the activations are already downsampled to 3D for the specified log_key"""
        assert (
            log_key in self.deeprepvizlogs
        ), f"the log_key {log_key} is not present in the loaded logs.\
    Currently present logs are {list(self.deeprepvizlogs.keys())}. Please load the log first using the self.load_log() method."
        return os.path.isfile(
            join(
                log_key,
                self.deeprepvizlogs[log_key]["checkpoints"][-1][0],
                self.deeprepvizlogs[log_key]["checkpoints"][-1][0],
                "tensors_3D.tsv",
            )
        )

    def downsample_activations(self, method=PCA(n_components=3), overwrite=False):
        """Perform dimensionality reduction on activations using the specified method.

        Args:
            method: A sklearn dimensionality reduction method (e.g.,
            sklearn.decomposition.PCA(), sklearn.manifold.TSNE()).

        Returns:
            numpy.ndarray: Activations reduced to 3D."""
        print(f"Reducing activations to 3D using {method.__class__.__name__}")
        for log_key, log_dict in self.deeprepvizlogs.items():
            if self._is_downsampled(log_key) and not overwrite:
                print(
                    f"The activations have already been reduced to 3D for {log_key}. Skipping ..."
                )
                continue
            else:
                for i, (ckpt_name, ckpt_data) in enumerate(
                    tqdm(log_dict["checkpoints"])
                ):
                    acts = ckpt_data["acts"]
                    method_fit = method.fit(acts)
                    acts_3D = method_fit.transform(acts)
                    ckpt_data["acts_3D"] = acts_3D
                    # save the acts_3D to the checkpoint folder as tensors_3D.tsv
                    np.savetxt(
                        join(log_key, ckpt_name, ckpt_name, "tensors_3D.tsv"),
                        acts_3D,
                        delimiter="\t",
                    )
                    assert (
                        acts_3D.shape[-1] == 3
                    ), f"the dim_reduct_method should be configured \
    to reduce the dimensions to 3 but it is currently reduced to {acts_3D.shape[-1]} from {acts.shape[-1]}"

    ###############################         Metrics        ########################################

    def get_metrics(self, log_key, ckpt_idx="best"):
        """Check if the metric is already computed for the specified checkpoint and return it"""
        assert (
            log_key in self.deeprepvizlogs
        ), f"the log_key {log_key} is not present in the loaded logs.\
 Currently present logs are {list(self.deeprepvizlogs.keys())}. Please load the log first using the self.load_log() method."
        if ckpt_idx == "best":
            ckpt_idx = self.deeprepvizlogs[log_key]["best_ckpt_idx"]

        ckpt_name, src_ckpt = self.deeprepvizlogs[log_key]["checkpoints"][ckpt_idx]
        if "act_metrics" in src_ckpt:
            return src_ckpt["act_metrics"]
        else:
            return None

    def compute_metrics(
        self, log_key, ckpt_idx="best", split="all", metrics=["dcor"], covariates="all"
    ):
        """Compute the specified metric between the activations and the covariates/confounds from self.conf_table"""

        metric_methods = {
            "con": compute_con_score,
            "r2": compute_r2_score,
            "costeta": compute_costeta_score,
            "mi": compute_mi_score,
            "dcor": compute_dcor_score,
        }

        assert (
            log_key in self.deeprepvizlogs
        ), f"the log_key {log_key} is not present in the loaded logs. Currently present logs are {list(self.deeprepvizlogs.keys())}. Please load the log first using the self.load_log() method."
        if ckpt_idx == "best":
            ckpt_idx = self.deeprepvizlogs[log_key]["best_ckpt_idx"]
        assert ckpt_idx >= 0 and ckpt_idx < len(
            self.deeprepvizlogs[log_key]["checkpoints"]
        ), f"the ckpt_idx should be between 0 and {len(deeprepvizlog['checkpoints'])-1} or 'best' but it is {ckpt_idx}"

        ckpt_name, src_ckpt = self.deeprepvizlogs[log_key]["checkpoints"][ckpt_idx]
        X = src_ckpt["acts"]

        if isinstance(covariates, str) and covariates == "all":
            covariates = [c for c in self.df_conf.columns if c not in [self.ID_col]]
        assert np.array(
            [(c in self.df_conf.columns) for c in covariates]
        ).all(), f"the covariates requested should be present in the conf_table but the following are not: {set(covariates) - set(self.df_conf.columns)}"

        results = {}
        existing_metrics = self.get_metrics(log_key, ckpt_idx)
        for metric in metrics:
            assert (
                metric in metric_methods
            ), f"Invalid metric. Available options are: {list(metric_methods.keys())}"
            metric_method = metric_methods[metric]

            if existing_metrics is not None and metric in existing_metrics:
                print(
                    f"the metric {metric} is already computed for ckpt_idx={ckpt_idx} from {log_key}. Overwriting..."
                )

            if self.debug:
                print(
                    f"generating {metric.upper()} values for n={len(covariates)} covariates using model representation at ckpt_idx={ckpt_idx} from {log_key}"
                )

            additional_kwargs = {}
            if metric in ["con", "r2", "costeta"]:
                additional_kwargs.update(
                    {
                        "y_pred_weights": np.array(src_ckpt[f"weights_0"]),
                        "y_pred_bias": np.array([src_ckpt[f"biases_0"]]),
                    }
                )

            result = dict(
                Parallel(n_jobs=1 if self.debug else -1)(
                    delayed(metric_method)(
                        X,
                        y=self.df_conf[covari].values,
                        pred_type=self.conf_dtypes[covari],
                        var_name=covari,
                        **additional_kwargs,
                    )
                    for covari in tqdm(covariates)
                )
            )

            results.update({metric: result})

        if "act_metrics" in src_ckpt:
            src_ckpt["act_metrics"].update(results)
        else:
            src_ckpt["act_metrics"] = results

        jsonfile = join(log_key, ckpt_name, ckpt_name, "metametadata.json")
        with open(jsonfile, "r") as fp:
            metametadata = json.load(fp)
            metametadata["act_metrics"] = src_ckpt["act_metrics"]
        with open(jsonfile, "w") as fp:
            json.dump(metametadata, fp, indent=4)

        return results

    def convert_log_to_v1_table(
        self,
        log_key,
        unique_name="",
        dim_reduct_method=PCA(n_components=3),
        overwrite=False,
    ):
        """ckpts_dir: path to the 'deeprepvizlog' foldercontaining all the checkpoint folders"""
        if not unique_name:
            unique_name = "default"
        # first save the ID and labels
        assert (
            log_key in self.deeprepvizlogs
        ), f"the log_key {log_key} is not present in the loaded logs.\
 Currently present logs are {list(self.deeprepvizlogs.keys())}. Please load the log first using the load_log() method."

        v1_table_path = f"{log_key}/DeepRepViz-v1-{unique_name}.csv"
        log_dict = self.deeprepvizlogs[log_key]
        IDs = log_dict["IDs"]
        labels = log_dict["labels"]
        df = pd.DataFrame(index=IDs)
        df["labels"] = labels
        # next, append with conf_table
        df.index.name = self.ID_col
        df = df.join(self.df_conf, on=self.ID_col)
        # change index name to what DeepRepViz v1 expects
        df.index.name = "subjectID"
        dfs = []
        # for all checkpoints, save the preds, acts and metrics
        for i, (ckpt_name, ckpt_data) in enumerate(tqdm(log_dict["checkpoints"])):
            ckpt_name = f"{unique_name}-{ckpt_name}"
            dfi = pd.DataFrame(index=IDs)
            dfi.index.name = "subjectID"
            # collect predicted logits and append
            preds = [
                (pred_col, vals)
                for pred_col, vals in ckpt_data.items()
                if "preds" in pred_col
            ]
            for pred_col, pred_vals in preds:
                i = pred_col.replace("preds_", "")
                dfi[f"Pred_{ckpt_name}_class{i}-logits"] = pred_vals
            # append activations
            acts = ckpt_data["acts"]
            if acts.shape[-1] == 3:
                acts_3D = acts
            elif "acts_3D" in ckpt_data:
                acts_3D = ckpt_data["acts_3D"]
            # if 3D activations are not computed yet, then compute them
            else:
                assert (
                    dim_reduct_method is not None
                ), "A dimensionality reduction method should be \
    provided in arg 'dim_reduct_method' since the activations are more than 3 dimensions (ex: sklearn.decomposition.PCA(n_components=3) or sklearn.manifold.TSNE(n_components=3))."
                self.downsample_activations(
                    method=dim_reduct_method, overwrite=overwrite
                )
                acts_3D = ckpt_data["acts_3D"]

            assert (
                acts_3D.shape[-1] == 3
            ), f"the dim_reduct_method should be configured to reduce the dimensions to 3 but it is currently {acts_3D.shape[-1]}"
            for i, suffix in enumerate(
                ["X", "Y", "Z"]
            ):  # TODO this is stupid, should be indexed by numbers and not letters
                dfi[f"Rep_{ckpt_name}_{suffix}"] = acts_3D[:, i]

            # append meta data
            for k, v in ckpt_data["metrics"].items():
                dfi[f"Meta_{ckpt_name}_Metric-{k}"] = [v] + [pd.NA] * (IDs.shape[0] - 1)

            # append weights and biases  #TODO v1 isnt ready for this yet
            # weights = [(col, vals) for col, vals in ckpt_data.items()  if 'weights' in col]
            # biases = [(col, vals) for col, vals in ckpt_data.items()  if 'biases' in col]
            # for k, (weight_col, weight_vals) in enumerate(weights):
            #     weight_vals = weight_vals
            #     i = (weight_col.replace('weights_',''))
            #     # if bias is present then append it at the end of the weights
            #     if len(biases)>0:
            #         bias_val = biases[k][1]
            #     if len(weight_vals) != 3:
            #         weights_3D = dim_reduct_method.transform(np.array(weight_vals).reshape(1,-1)).squeeze().tolist()
            #     else:
            #         weights_3D = weight_vals

            #     dfi[f'Vec_{ckpt_name}_-class{i}-Weights'] = weights_3D + [bias_val] + [pd.NA]*(IDs.shape[0]-len(weights_3D)-1)

            dfs.append(dfi)

        df = pd.concat([df, *dfs], axis=1)

        # save
        if overwrite or not os.path.isfile(v1_table_path):
            df.to_csv(v1_table_path)
        else:  # if the file already exists then throw a warning
            warnings.warn(
                f"The file {v1_table_path} already exists. Skipping saving the table."
            )
        return df

def compute_dcor_score(X, y, var_name="", pred_type="classif"):
    y = y.reshape(-1, 1)
    assert (
        X.shape[0] == y.shape[0]
    ), f"the shape of the activations and covariates should be (N, D) and (N, 1)\
 respectively but they are {X.shape} and {y.shape} respectively"
    dcor_score = float(dcor.distance_correlation_sqr(X, y))
    return var_name, dcor_score

def compute_mi_score(X, y, var_name, pred_type="classif"):
    if pred_type in ["classif_binary", "classif"]:
        mi = mutual_info_classif(X, y.ravel())
    else:
        mi = mutual_info_regression(X, y.ravel())
    # print(var_name, mi, np.max(mi), np.mean(mi), np.median(mi), np.min(mi))
    return var_name, float(
        np.mean(mi)
    )  # TODO find another metric that gives a single value of MI rather than 1 value per feature

# TODO make these a single class
def compute_con_score(X, y, var_name, pred_type, y_pred_weights, y_pred_bias):
    con, (r2, cos_teta, model) = con_score(
        X,
        y,
        y_pred_weights=y_pred_weights,
        y_pred_bias=y_pred_bias,
        pred_type=pred_type,
        return_everything=True,
    )
    return var_name, con

def compute_r2_score(X, y, var_name, pred_type, y_pred_weights, y_pred_bias):
    _, (r2, cos_teta, model) = con_score(
        X,
        y,
        y_pred_weights=y_pred_weights,
        y_pred_bias=y_pred_bias,
        pred_type=pred_type,
        return_everything=True,
    )
    return var_name, r2

def compute_costeta_score(X, y, var_name, pred_type, y_pred_weights, y_pred_bias):
    _, (r2, cos_teta, model) = con_score(
        X,
        y,
        y_pred_weights=y_pred_weights,
        y_pred_bias=y_pred_bias,
        pred_type=pred_type,
        return_everything=True,
    )
    return var_name, cos_teta

#########################      start of pseudo r-squared implementations     ################################


def full_log_likelihood(w, X, y):
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)


def null_log_likelihood(w, X, y):
    z = np.array(
        [w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]
    ).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))


# McKelvey-&-Zavoina-R^2
def mz_rsquare(w, X, y):
    # if weights have intercept then add additional column to X
    if X.shape[-1] + 1 == w.shape[0]:
        X = np.append(X, np.zeros(shape=(X.shape[0], 1)), axis=-1)
    logits = np.dot(X, w).reshape(1, X.shape[0])
    return np.var(logits) / (np.var(logits) + (np.power(np.pi, 2.0) / 3.0))


#########################         Helper functions            ################################################
def _compute_d2(pseudo_r2_type, c_pred_weights, X, conf):
    if pseudo_r2_type == "mcf":
        return mcfadden_adjusted_rsquare(c_pred_weights.T, X, conf)
    elif pseudo_r2_type == "mz":
        return mz_rsquare(c_pred_weights.T, X, conf)
    else:
        raise ValueError(
            f"pseudo_r2_type={pseudo_r2_type} is invalid. Currently supported types are ['mz', 'mcf']"
        )


def _unit_vector(vector):
    return vector / (np.linalg.norm(vector) + sys.float_info.epsilon)


def _get_cos_teta(v1, v2):
    # scipy.spatial.distance
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def _compute_con_score(r2, c_pred_weights, y_pred_weights):
    c_pred_weights = c_pred_weights.squeeze()
    y_pred_weights = y_pred_weights.squeeze()
    assert (
        c_pred_weights.shape == y_pred_weights.shape
    ), f"c_pred_weights.shape {c_pred_weights.shape} != y_pred_weights.shape {y_pred_weights.shape}"
    cos_teta = _get_cos_teta(c_pred_weights, y_pred_weights)

    con_score = r2 * abs(cos_teta)
    con_score = 0.0 if con_score < 0 else con_score
    return con_score, cos_teta


#########################         Con Score            ################################################
def con_score(
    X,
    conf,
    y_pred_weights,
    y_pred_bias,
    pred_type="classif",
    pseudo_r2_type="mz",
    use_statsmodels=False,
    return_everything=False,
):
    """
    pred_type: can be either ['classif', 'classif_binary', 'regression']. Defines if
        X->conf should be modelled using LogisticRegression or LinearRegression.
    y_pred_bias: If set to None it is assumed that the model was trained without a bias
    pseudo_r2_type: which algorithm to use to calculate the pseudo R2 (or D2).
        Currently supported are ['mz', 'mcf'] that implement McKelvey & Zavoina's R2 and McFadden's adjusted R2 resp.
        See https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-what-are-pseudo-r-squareds/
        only applicable when pred_type='classif' or 'classif_binary'
    use_statsmodels: use statsmodels package to fit X->conf instead of sklearn models
    """
    # prepare
    pred_type = pred_type.lower()
    assert pred_type in [
        "classif_binary",
        "classif",
        "regression",
    ], f"\
Currently supported value for pred_type are ['classif_binary', 'classif', 'regression'] but given {pred_type}."
    assert pseudo_r2_type in [
        "mz",
        "mcf",
    ], f"\
Currently supported value for pseudo_r2_type are ['mz', 'mcf'] but given {pseudo_r2_type}."
    # join y_pred bias with weights
    if y_pred_bias is not None:
        # reshape to 2D arrays before concatenating
        y_pred_bias = y_pred_bias.reshape(-1, 1)
        if y_pred_weights.ndim < 2:
            y_pred_weights = y_pred_weights.reshape(1, -1)
        # concatenate on the last axis
        y_pred_weights = np.concatenate([y_pred_weights, y_pred_bias], axis=-1)
        if use_statsmodels:
            X_int = sm.add_constant(X, prepend=False)

    # Perform X->conf logistic regression
    if pred_type == "classif_binary" or pred_type == "classif":
        # if c is continuous but y was categorical then pseudo-categorize c and do classification
        # if len(np.unique(conf))>2: # TODO remove hardcoded check
        #     print(f"[WARN] pseudo-categorized the continuous conf variable ({len(np.unique(conf))} unique values) to binary using it's mean.")
        #     conf = (conf>conf.mean()).astype(int)

        if use_statsmodels:
            model = sm.MNLogit(conf, X_int).fit(disp=False)
            r2 = model.prsquared
            c_pred_weights = model.params
            if c_pred_weights.ndim < 2:
                c_pred_weights = c_pred_weights.reshape(1, -1)
        else:
            # use sklearn
            model = LogisticRegression(
                fit_intercept=(y_pred_bias is not None),
                solver="lbfgs",
                penalty=None,
                max_iter=1000,
            )
            model.fit(X, conf)
            c_pred_weights = np.array(model.coef_).squeeze()
            # always force weight vectors to 2D to make it harmonize with multiclass weight dims (2D)
            if c_pred_weights.ndim < 2:
                c_pred_weights = c_pred_weights.reshape(1, -1)
            # append intercept to end of weights if enabled
            if y_pred_bias is not None:
                c_pred_bias = model.intercept_.squeeze().reshape(-1, 1)
                # print('[D]',c_pred_weights.shape, c_pred_bias.shape)
                c_pred_weights = np.concatenate([c_pred_weights, c_pred_bias], axis=-1)

            if c_pred_weights.shape[0] == 1:
                # calculate D2 using selected method
                r2 = _compute_d2(pseudo_r2_type, c_pred_weights, X, conf)

            # if conf is multiclass classification then compute R2 individually for each class
            elif c_pred_weights.shape[0] > 1:
                r2s = []
                for i in range(c_pred_weights.shape[0]):
                    r2 = _compute_d2(pseudo_r2_type, c_pred_weights[i, :], X, conf)
                    r2s.append(r2)
                # print('[D]', r2s, np.unique(conf, return_counts))
            else:
                raise ValueError(
                    f"c_pred_weights.shape={c_pred_weights.shape} is invalid"
                )

    # Perform X->conf linear regression
    elif pred_type == "regression":
        if use_statsmodels:
            model = sm.OLS(conf, X_int).fit()
            r2 = model.rsquared_adj
            c_pred_weights = model.params
        else:  # use sklearn
            model = LinearRegression(fit_intercept=(y_pred_bias is not None))
            model.fit(X, conf)
            c_pred_weights = np.array(model.coef_).squeeze()
            if c_pred_weights.ndim < 2:
                c_pred_weights = c_pred_weights.reshape(1, -1)
            # append intercept to end of weights if enabled
            if y_pred_bias is not None:
                c_pred_bias = model.intercept_.squeeze().reshape(-1, 1)
                if c_pred_weights.ndim < 2:
                    c_pred_weights = c_pred_weights.reshape(1, -1)
                c_pred_weights = np.concatenate([c_pred_weights, c_pred_bias], axis=-1)
            # calculate R2 using the sklearn method
            r2 = model.score(X, conf)
        # for regression apply sigmoid on the weight vector to make cos(theta) valid
        # pred_type = align(y_pred_weights)
        # c_pred_weights = align(c_pred_weights)

    # if y is multiclass then iterate over the each class of y and return all scores as a list
    con_scores_yi = []
    for yi in range(y_pred_weights.shape[0]):  # if y_pred_weights.shape[0]>1:
        y_pred_weights_yi = y_pred_weights[yi, :]
        # if conf is multiclass classification then iterate over the each class and return the highest conf
        if c_pred_weights.shape[0] > 1:
            con_score, cos_teta = 0, None
            for ci in range(c_pred_weights.shape[0]):
                con_score_ci, cos_teta_ci = _compute_con_score(
                    r2s[ci], c_pred_weights[ci, :], y_pred_weights_yi
                )
                if con_score_ci > con_score:
                    best_class, con_score, cos_teta = ci, con_score_ci, cos_teta_ci
        else:
            # if conf is binary classification/regression
            con_score, cos_teta = _compute_con_score(
                r2, c_pred_weights, y_pred_weights_yi
            )

        con_scores_yi.append(con_score)

        if len(con_scores_yi) == 1:
            con_score = con_scores_yi[0]  # if y_pred_weights.shape[0]==1
        else:
            con_score = con_scores_yi  # if y_pred_weights.shape[0]>1

    if return_everything:
        return con_score, (r2, cos_teta, model)
    else:
        return con_score