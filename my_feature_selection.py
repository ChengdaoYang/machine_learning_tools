import numbers

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection._base import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score, GridSearchCV


class AutoSequentialFeatureSelector(SequentialFeatureSelector):
    """
    Modified SequentialFeatureSelector auto add/remove feature as long as it 
    would improve model performance, which eval based on a cross validation and
    GridSearch optimiazed hyper params set. (much more expensive than the 
    paren SequentialFeatureSelector

    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.

    grid_cv_param: dict
        A potential hyper parameter space for GridSearchCV

    rest params: see parent SequentialFeatureSelector


    Attributes
    ----------
    feature_trainning_scores_ : pandas df 
        each row show the feature mask and its scores in each search iter

    rest attr: see parent SequentialFeatureSelector
    """

    def __init__(
        self,
        estimator,
        grid_cv_param,
        *,
        opt_speed=False,
        n_features_to_select=1.,
        direction="forward",
        scoring=None,
        cv=3,
        n_jobs=None,
    ):
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        self._grid_cv_params = grid_cv_param
        self._feature_idxs = list() 
        self._feature_scores= list()  
        self._opt_speed = opt_speed  # Speed up by stop when no improve score


    @property
    def opt_speed(self):
        return self._opt_speed

    @opt_speed.setter
    def opt_speed(self, opt_speed):
        if isinstance(opt_speed, bool):
            self._opt_speed = opt_speed
        else:
            raise TypeError(
                 "opt_speed option take boolean,"
                f" not {type(opt_speed)}"
            )

    
    @property
    def feature_trainning_scores_(self):
        return pd.DataFrame(
            zip(
                self._feature_idxs,
                self._feature_scores
            )
        ).rename(
            columns={0: "feature_idx", 1: "feature_score"}
        )


    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]

        error_msg = (
            "n_features_to_select must be either None, an "
            "integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )
        if self.n_features_to_select is None:
            self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        if self.direction not in ("forward", "backward"):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        trainning_mask = np.zeros_like(current_mask, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.direction == "forward"
            else n_features - self.n_features_to_select_
        )
        current_best_score = float('-inf')
        for _ in range(n_iterations):
            new_feature_idx, new_feature_score = self._get_best_new_feature(
                cloned_estimator,
                X,
                y,
                trainning_mask  # Instead of current_mask 
            )
            # Record Trainning process
            trainning_mask[new_feature_idx] = True
            self._feature_idxs.append(tuple(trainning_mask))
            self._feature_scores.append(new_feature_score)

            if new_feature_score > current_best_score:
                current_best_score = new_feature_score
                current_mask[new_feature_idx] = True
            elif self._opt_speed is True:
                break
            else:
                pass

        if self.direction == "backward":
            current_mask = ~current_mask
        self.support_ = current_mask

        return self

    def _get_best_new_feature(self, estimator, X, y, current_mask):
        # Return the best new feature to add to the current_mask and its scores
        # i.e. return the best new feature to add (resp. remove) when doing 
        # forward selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            X_new = X[:, candidate_mask]
            scores[feature_idx] = GridSearchCV(
                estimator,
                self._grid_cv_params,
                cv=self.cv,
            ).fit(
                X_new,
                y
            ).best_score_
        best_feature_idx =  max(
            scores,
            key=lambda feature_idx: scores[feature_idx]
        )
        best_feature_score = scores.get(best_feature_idx)
        return best_feature_idx, best_feature_score

