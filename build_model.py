# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np

from typing import List, Tuple, Callable, Dict, Optional
from collections import Counter
from copy import deepcopy
from time import perf_counter
import sys

from cheminfo_toolkit import Molecule, Fingerprint_engine

from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def select_groups(molecules: List[Molecule], minimum_group_size: int=10,
                  small_groups_as_negative=True, pulled_small_group_name: str="miscellaneous chemistry")-> List[Molecule]:
    '''
    Keeps groups with at least minimum_group_size structures. Smaller groups are either ignored or put into a group
    :param molecules: List of Molecule objects
    :param minimum_group_size: groups with at least as many processable structures will keep their name
    :param small_groups_as_negative: if True, small groups will be pulled together into a group named pulled_small_group_name
    :param pulled_small_group_name: name given to the group comprising the structures from the small groups. It is ignored
    if small_groups_as_negative is False
    :return: List of molecules with the group names adjusted
    '''

    # create a deep copy not to destroy the original molecule list
    molecules = deepcopy(molecules)

    # select the groups for which we have a sufficient number of processable structures
    counts = Counter([molecule.group for molecule in molecules])
    large_groups = [group for group in counts if counts[group]>=minimum_group_size]
    n_subs_large_groups = len(list(filter(lambda mol: mol.group in large_groups, molecules)))
    log.info(f'from the initial {len(counts)} groups, {len(large_groups)} groups are large and contain at least {minimum_group_size} structures')
    log.info(f'the large groups have {n_subs_large_groups} structures from a total of {len(molecules)}')
    n_subs_small_groups = len(counts)-len(large_groups)
    if n_subs_small_groups>0:
        log.info(f'from the initial {len(counts)} groups, {len(counts)-len(large_groups)} groups are small and contain fewer than {minimum_group_size} structures')
        log.info(f'the small groups have {len(molecules)-n_subs_large_groups} structures from a total of {len(molecules)}')

    # coallesce the small groups
    if small_groups_as_negative and n_subs_small_groups>0:
        small_groups = [group for group in counts if counts[group]<minimum_group_size]
        for mol in molecules:
            if mol.group in small_groups:
                mol.group = pulled_small_group_name
        log.info(f'small groups were coalesced into the group {pulled_small_group_name}')
    elif not small_groups_as_negative and n_subs_small_groups>0:
        log.info(f'small groups will be ignored')
    return molecules



def split_molecules_train_test(molecules: List[Molecule], test_size=0.2,
                               random_state: int=0, stratify: bool=True)->Tuple[List[Molecule], List[Molecule]]:
    '''
    Splits the molecule collections to train and test set
    :param molecules: List of molecules in the dataset
    :param test_size: fraction of molecules to leave in the test set
    :param random_state: seed to ensure deterministic behaviour
    :param stratify: if True, the split is stratifed using the group names
    :return: Tuple with training and test set molecules as lists
    '''

    if stratify:
        log.info('train/test split will use stratification')
        y = [mol.group for mol in molecules]
        molecules_train, molecules_test = train_test_split(molecules, test_size=test_size, random_state=random_state, stratify=y)
        log.info(f'the training set has {len(molecules_train)} molecules, the test set has {len(molecules_test)} molecules')
    else:
        log.info('train/test split will not use stratification')
        molecules_train, molecules_test = train_test_split(molecules, test_size=test_size, random_state=random_state, stratify=None)
        log.info(f'the training set has {len(molecules_train)} molecules, the test set has {len(molecules_test)} molecules')
    return molecules_train, molecules_test



def build_gradient_boosting_classifier(molecules_train: List[Molecule],
                                       fingerprint_engine: Fingerprint_engine,
                                       random_state: int=0)-> Dict:
    '''
    Build a gradient boosting multiclass classifier to predict the group membership.

    :param molecules_train: List of molecules to be used as a training set
    :param fingerprint_engine: Fingerprint_engine to be used for fingerprint generation
    :param random_state: seed to ensure deterministic behaviour
    :return: Tuple with
             - the function that receives a molecule and predicts the group it belongs to; the function has embedded
               the fingerprint engine for consistency with the model building
             - dictionary with model details

    '''
    # set the fingerprints as features and the group names as target
    X_train = np.array([np.array(mol.compute_fingerprint(fingerprint_engine)) for mol in molecules_train])
    y_train = np.array([mol.group for mol in molecules_train])

    # initiate the pipeline
    pipeline = Pipeline(steps=[])

    # .. remove features with zero variance
    sel = VarianceThreshold(threshold=0.)
    pipeline.steps.append(('zero_variance_feature_elimination', sel))
    log.info(f'after removing zero variance features the dataset shape changed from {X_train.shape} to {pipeline.fit_transform(X_train).shape}')

    # gradient boosting multiclass classfier
    min_samples_split = 2 # included in grid search; the minimum number of samples required to split an internal node, 2 means fully developed trees
    max_leaf_nodes = None # default; grow trees with max_leaf_nodes
    max_depth = 3 # default, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    learning_rate = 0.1
    min_samples_leaf = 1 # default; the minimum number of samples required to be at a leaf node
    n_estimators = 200 # included in grid search; the number of trees in the forest
    max_features=0.3 # included in grid search; the number of features to consider when looking for the best split
    gb = GradientBoostingClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth, learning_rate=0.1,
                                    min_samples_split=min_samples_split, random_state=random_state, max_features=max_features)
    pipeline.steps.append(('gb', gb))

    # cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    # scoring function
    scorer = lambda estimator, X, y: f1_score(y, estimator.predict(X), average='macro') # equivalent to scorer='f1_macro' but more flexible, https://scikit-learn.org/stable/modules/cross_validation.html

    # grid search, hyper parameter tuning
    # parameters = {'gb__n_estimators': [50, 100, 150, 200, 250, 300],
    #               'gb__max_features': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.],
    #               'gb__min_samples_split': [2, 3, 4]}
    # parameters = {'gb__n_estimators': [50, 100, 200],
    #               'gb__learning_rate': [0.02, 0.05, 0.1, 0.2],
    #               'gb__max_depth': [2, 3, 5, 8, 10]}
    parameters = {'gb__n_estimators': [50, 100, 200],
                  'gb__learning_rate': [0.02, 0.05, 0.1, 0.2],
                  'gb__min_samples_split': [2, 3, 4],
                  'gb__max_features': [0.002, 'sqrt', 'log2', 0.5],
                  'gb__max_depth': [2, 3, 5, None]}
    # parameters = {'gb__n_estimators': [10, 20],
    #               'gb__max_features': [0.5],
    #               'gb__min_samples_split': [3]}


    clf = GridSearchCV(pipeline, parameters, cv=skf, scoring=scorer, n_jobs=6, refit=True, verbose=10)
    time_start = perf_counter()
    res = clf.fit(X_train, y_train)
    log.info(f'grid search took {perf_counter()-time_start:0.2f} sec (wall time)')

    # assemble grid search results
    grid_search_results = pd.DataFrame(res.cv_results_)

    # report timing
    average_fit_time = res.cv_results_['mean_fit_time'].mean()
    log.info(f'the average fit time was {average_fit_time:0.2f} sec')
    log.info(f'the average fit time per cross-validation was {n_splits*average_fit_time:0.2f} sec')
    log.info(f"the total fit time for all cross-validations was {n_splits*res.cv_results_['mean_fit_time'].sum():0.2f} sec")

    # best parameters
    log.info(f'best parameters are {res.best_params_}')

    # scoring of best model during cross validation
    best_estimator = res.best_estimator_
    log.info(f'best mean cross-validation score is {res.best_score_:0.3f}')
    best_attempt = (grid_search_results
                   .loc[grid_search_results['rank_test_score']==1]
                   .filter(regex=r'split\d+_test_score', axis=1)
                   .applymap('{:,.3f}'.format)
                   .squeeze()
                   .to_list())
    log.info(f'  .. that is the average of the fold scores {", ".join(best_attempt)}')
    log.info(f'worst mean cross-validation score is {res.cv_results_["mean_test_score"].min():0.3f}')
    log.info(f'average mean cross-validation score is {res.cv_results_["mean_test_score"].mean():0.3f}')

    # scoring of best model for training and test set
    log.info(f'score using train set {scorer(best_estimator, X_train, y_train):0.3f}')

    # return the model details for post-processing
    model_details = {}
    model_details['best mean cross-validation score'] = res.best_score_
    model_details['best estimator'] = best_estimator
    model_details['grid search results'] = grid_search_results
    model_details['fingerprint engine'] = fingerprint_engine

    return model_details


def build_random_forest_classifier(molecules_train: List[Molecule],
                                   fingerprint_engine: Fingerprint_engine,
                                   random_state: int=0,
                                   parameters: Optional[dict] = None)-> dict:
    '''
    Build a random forest multiclass classifier to predict the group membership.

    :param molecules_train: List of molecules to be used as a training set
    :param fingerprint_engine: Fingerprint_engine to be used for fingerprint generation
    :param random_state: seed to ensure deterministic behaviour
    :param parameters: parameters for the model grid search, if None it uses defaults
    :return: Tuple with
             - the function that receives a molecule and predicts the group it belongs to; the function has embedded
                the fingerprint engine for consistency with the model building
             - dictionary with model details

    '''

    # set the fingerprints as features and the group names as target
    X_train = np.array([np.array(mol.compute_fingerprint(fingerprint_engine)) for mol in molecules_train])
    y_train = np.array([mol.group for mol in molecules_train])

    # initiate the pipeline
    pipeline = Pipeline(steps=[])

    # .. remove features with zero variance
    sel = VarianceThreshold(threshold=0.)
    pipeline.steps.append(('zero_variance_feature_elimination', sel))
    log.info(f'after removing zero variance features the dataset shape changed from {X_train.shape} to {pipeline.fit_transform(X_train).shape}')

    # random forest multiclass classifier
    min_samples_split = 2 # included in grid search; the minimum number of samples required to split an internal node, 2 means fully developed trees
    max_leaf_nodes = None # default; grow trees with max_leaf_nodes
    max_depth = None # default, The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_leaf = 1 # default; the minimum number of samples required to be at a leaf node
    n_estimators = 200 # included in grid search; the number of trees in the forest
    max_features=0.3 # included in grid search; the number of features to consider when looking for the best split
    rf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                                min_samples_split=min_samples_split, random_state=random_state, max_features=max_features)
    pipeline.steps.append(('rf', rf))


    # cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    # scoring function
    scorer = lambda estimator, X, y: f1_score(y, estimator.predict(X), average='macro') # equivalent to scorer='f1_macro' but more flexible, https://scikit-learn.org/stable/modules/cross_validation.html

    # grid search, hyper parameter tuning
    if parameters is None:
        parameters = {'rf__n_estimators': [50, 100, 150, 200, 250, 300],
                      'rf__max_features': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.],
                      'rf__min_samples_split': [2, 3, 4]}
        # parameters = {'rf__n_estimators': [150],
        #               'rf__max_features': [0.01, 0.02],
        #               'rf__min_samples_split': [3]}

    clf = GridSearchCV(pipeline, parameters, cv=skf, scoring=scorer, n_jobs=6, refit=True, verbose=10)
    time_start = perf_counter()
    res = clf.fit(X_train, y_train)
    log.info(f'grid search took {perf_counter()-time_start:0.2f} sec (wall time)')

    # assemble grid search results
    grid_search_results = pd.DataFrame(res.cv_results_)

    # report timing
    average_fit_time = res.cv_results_['mean_fit_time'].mean()
    log.info(f'the average fit time was {average_fit_time:0.2f} sec')
    log.info(f'the average fit time per cross-validation was {n_splits*average_fit_time:0.2f} sec')
    log.info(f"the total fit time for all cross-validations was {n_splits*res.cv_results_['mean_fit_time'].sum():0.2f} sec")

    # best parameters
    log.info(f'best parameters are {res.best_params_}')

    # scoring of best model during cross validation
    best_estimator = res.best_estimator_
    log.info(f'best mean cross-validation score is {res.best_score_:0.3f}')
    best_attempt = (grid_search_results
                   .loc[grid_search_results['rank_test_score']==1]
                   .filter(regex=r'split\d+_test_score', axis=1)
                   .applymap('{:,.3f}'.format)
                   .squeeze()
                   .to_list())
    log.info(f'  .. that is the average of the fold scores {", ".join(best_attempt)}')
    log.info(f'worst mean cross-validation score is {res.cv_results_["mean_test_score"].min():0.3f}')
    log.info(f'average mean cross-validation score is {res.cv_results_["mean_test_score"].mean():0.3f}')

    # scoring of best model for training and test set
    log.info(f'score using train set {scorer(best_estimator, X_train, y_train):0.3f}')

    # return the model details for post-processing
    model_details = {}
    model_details['best mean cross-validation score'] = res.best_score_
    model_details['best estimator'] = best_estimator
    model_details['grid search results'] = grid_search_results
    model_details['fingerprint engine'] = fingerprint_engine

    return model_details


# create predictor function for a random forest model
def group_predictor_rf(molecule: Molecule, model_details: Dict, all_groups=False)-> str:
    '''
    Function that accepts a molecule and a random forest estimator and predicts the group it belongs to
    :param Molecule: input molecule
    :param model_details: dictionary with model details, including the best estimator and fingerprint engine
    :param all_groups: boolean, if True then the function returns a dictionary with all group probabilities
                       if False only the name of the group of highest probability is returned
    :return: group name of highest probability or dictionary with group probabilities
    '''

    fingerprint_engine = model_details['fingerprint engine']
    best_estimator = model_details['best estimator']
    X = np.array(molecule.compute_fingerprint(fingerprint_engine)).reshape(1, -1)
    if all_groups:
        probabilities = dict(zip(best_estimator.classes_, best_estimator.predict_proba(X).ravel()))
        return probabilities
    else:
        return best_estimator.predict(X)[0]





def build_kNeighbours_classifier(molecules_train: List[Molecule],
                                 fingerprint_engine: Fingerprint_engine,
                                 random_state: int=0)-> Dict:
    '''
    Build a k nearest neighbours classifier to predict the group membership.

    :param molecules_train: List of molecules to be used as a training set
    :param fingerprint_engine: Fingerprint_engine to be used for fingerprint generation
    :param random_state: seed to ensure deterministic behaviour
    :return: Tuple with
             - the function that receives a molecule and predicts the group it belongs to; the function has embedded
                the fingerprint engine for consistency with the model building
             - dictionary with model details

    '''

    # set the fingerprints as features and the group names as target
    X_train = np.array([np.array(mol.compute_fingerprint(fingerprint_engine)) for mol in molecules_train]).astype(bool)
    y_train = np.array([mol.group for mol in molecules_train])

    # initiate the pipeline
    pipeline = Pipeline(steps=[])

    # .. remove features with zero variance
    sel = VarianceThreshold(threshold=0.)
    pipeline.steps.append(('zero_variance_feature_elimination', sel))
    log.info(f'after removing zero variance features the dataset shape changed from {X_train.shape} to {pipeline.fit_transform(X_train).shape}')

    # scoring function
    scorer = lambda estimator, X, y: f1_score(y, estimator.predict(X), average='macro') # equivalent to scorer='f1_macro' but more flexible, https://scikit-learn.org/stable/modules/cross_validation.html

    # nearest neighbours classifier
    n_neighbors = 5 # included in grid search
    weights = 'uniform' # included in grid search
    kn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='jaccard')
    pipeline.steps.append(('kn', kn))

    # cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    # scoring function
    scorer = lambda estimator, X, y: f1_score(y, estimator.predict(X),
                                              average='macro')  # equivalent to scorer='f1_macro' but more flexible, https://scikit-learn.org/stable/modules/cross_validation.html

    # grid search, hyper parameter tuning
    parameters = {'kn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'kn__weights': ['uniform', 'distance']
                  }


    clf = GridSearchCV(pipeline, parameters, cv=skf, scoring=scorer, n_jobs=6, refit=True, verbose=10)
    time_start = perf_counter()
    res = clf.fit(X_train, y_train)
    log.info(f'grid search took {perf_counter()-time_start:0.2f} sec (wall time)')



    # assemble grid search results
    grid_search_results = pd.DataFrame(res.cv_results_)

    # report timing
    average_fit_time = res.cv_results_['mean_fit_time'].mean()
    log.info(f'the average fit time was {average_fit_time:0.2f} sec')
    log.info(f'the average fit time per cross-validation was {n_splits*average_fit_time:0.2f} sec')
    log.info(f"the total fit time for all cross-validations was {n_splits*res.cv_results_['mean_fit_time'].sum():0.2f} sec")

    # best parameters
    log.info(f'best parameters are {res.best_params_}')

    # scoring of best model during cross validation
    best_estimator = res.best_estimator_
    log.info(f'best mean cross-validation score is {res.best_score_:0.3f}')
    best_attempt = (grid_search_results
                   .loc[grid_search_results['rank_test_score']==1]
                   .filter(regex=r'split\d+_test_score', axis=1)
                   .head(1)
                   .applymap('{:,.3f}'.format)
                   .squeeze()
                   .to_list())
    log.info(f'  .. that is the average of the fold scores {", ".join(best_attempt)}')
    log.info(f'worst mean cross-validation score is {res.cv_results_["mean_test_score"].min():0.3f}')
    log.info(f'average mean cross-validation score is {res.cv_results_["mean_test_score"].mean():0.3f}')

    # scoring of best model for training and test set
    log.info(f'score using train set {scorer(best_estimator, X_train, y_train):0.3f}')


    # return the model details for post-processing
    model_details = {}
    model_details['best mean cross-validation score'] = res.best_score_
    model_details['best estimator'] = best_estimator
    model_details['grid search results'] = grid_search_results
    model_details['fingerprint engine'] = fingerprint_engine

    return model_details



# create predictor function
def group_predictor_kn(molecule: Molecule, model_details: Dict,)-> str:
    '''
    Function that accepts a molecule and predicts the group it belongs to
    :param Molecule: input molecule
    :param model_details: dictionary with model details, including the best estimator and fingerprint engine
    :return: group name
    '''

    fingerprint_engine = model_details['fingerprint engine']
    best_estimator = model_details['best estimator']
    X = np.array(molecule.compute_fingerprint(fingerprint_engine)).reshape(1, -1).astype(bool)
    return best_estimator.predict(X)[0]