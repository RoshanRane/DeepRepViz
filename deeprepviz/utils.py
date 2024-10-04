import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from scipy.special import softmax, expit
import torch
from torchmetrics import Metric


class D2metric(Metric):
    def __init__(self, num_classes=2):
        super().__init__()
        self.add_state("targets", default=torch.Tensor([]))
        self.add_state("logits",  default=torch.Tensor([]))
        self.unique_y = list(range(num_classes))

    def update(self, logit: torch.Tensor, target: torch.Tensor):
        assert len(target) == len(logit), f"target.shape={target.shape} but logits.shape={logit.shape}"
        # if setting for the first time
        if len(self.targets)==0:
            self.targets = target
            self.logits = logit
        else:
            self.targets = torch.cat([self.targets,target], dim=0)
            self.logits  = torch.cat([self.logits, logit ], dim=0)

    def compute(self):
        return explained_deviance(
            self.targets.detach().cpu(),
            y_pred_logits=self.logits.detach().cpu(),
            unique_y=self.unique_y)

def explained_deviance(y_true, y_pred_logits=None, y_pred_probas=None,
                       returnloglikes=False, unique_y=[0,1]):
    """Computes explained_deviance score to be comparable to explained_variance
    Function taken from https://github.com/RoshanRane/Deviance_explained/blob/main/deviance.py"""

    assert y_pred_logits is not None or y_pred_probas is not None, "Either the predicted probabilities \
(y_pred_probas) or the predicted logit values (y_pred_logits) should be provided. But neither of the two were provided."

    if y_pred_logits is not None and y_pred_probas is None:
        # check if binary or multiclass classification
        if y_pred_logits.ndim == 1:
            y_pred_probas = expit(y_pred_logits)
        elif y_pred_logits.ndim == 2:
            y_pred_probas = softmax(y_pred_logits, axis=-1)
        else: # invalid
            raise ValueError(f"logits passed seem to have incorrect shape of {y_pred_logits.shape}")

    if y_pred_probas.ndim == 1: y_pred_probas = np.stack([1-y_pred_probas, y_pred_probas], axis=-1)
    total_probas = y_pred_probas.sum(axis=-1).round(decimals=4)
    assert (abs(total_probas-1.)<0.1).all(), f"the probabilities do not sum to one, {total_probas}"
    unique_y = np.unique(y_true)
    # compute a null model's predicted probability
    X_dummy = np.zeros(len(y_true))
    y_null_probas = DummyClassifier(strategy='prior').fit(X_dummy, y_true).predict_proba(X_dummy)
    #strategy : {"most_frequent", "prior", "stratified", "uniform",  "constant"}
    # suggestion from https://stackoverflow.com/a/53215317
    llf = -log_loss(y_true, y_pred_probas, normalize=False, labels=[0,1]) # unique_y TODO remove hardcoding
    llnull = -log_loss(y_true, y_null_probas, normalize=False, labels=[0,1]) # unique_y TODO remove hardcoding
    ### McFadden’s pseudo-R-squared: 1 - (llf / llnull)
    explained_deviance = 1 - (llf / llnull)
    ## Cox & Snell’s pseudo-R-squared: 1 - exp((llnull - llf)*(2/nobs))
    # explained_deviance = 1 - np.exp((llnull - llf) * (2 / len(y_pred_probas))) ## TODO, not implemented
    if returnloglikes:
        return explained_deviance, {'loglike_model':llf, 'loglike_null':llnull}
    else:
        return explained_deviance