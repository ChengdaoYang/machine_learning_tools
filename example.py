import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV

from my_feature_selection import AutoSequentialFeatureSelector


if __name__ == "__main__":
    # Load data
    X, y = load_iris(return_X_y=True)
    # Add random feature noise
    X = np.c_[np.random.rand(*X.shape), X, np.random.rand(*X.shape)]
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25)
    # Create hyper parameter set for Gridsearch 
    k_range = list(range(1, 31))
    knn_grid_params = dict(n_neighbors=k_range)
    svc_grid_params = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    # Create Feature selector obj
    asfs_svc = AutoSequentialFeatureSelector(
        SVC(),
        svc_grid_params,
    )
    
    # Build feature select  & training pipeline
    asfs_knn_pipe = Pipeline([
         ('feature_selection',  asfs_svc),
           ('classification', SVC())
           ])
    asfs_knn_pipe.fit(X_train, y_train)


    # standard seq select and train pipline
    sfs = SequentialFeatureSelector(SVC(), n_features_to_select=1.)
    sfs_knn_pipe = Pipeline([
         ('feature_selection',  sfs),
           ('classification', SVC())
           ])
    sfs_knn_pipe.fit(X_train, y_train)

    # No feature selection
    no_sfs = GridSearchCV(
        SVC(),
        svc_grid_params,
        cv=3,
    ).fit(
        X_train,
        y_train
    )

    # Compare results
    print("asfs: ",asfs_knn_pipe.score(X_test, y_test))
    print("sfs: ",sfs_knn_pipe.score(X_test, y_test))
    print("no sfs: ",no_sfs.score(X_test, y_test))
    #print(asfs_svc.feature_trainning_scores_)
    # print(cross_val_score(asfs_knn_pipe, X_test, y_test, cv=3))
    # print(cross_val_score(sfs_knn_pipe, X_test, y_test, cv=3))
