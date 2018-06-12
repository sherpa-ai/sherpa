from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import time
import sherpa

parameters = [sherpa.Discrete('n_estimators', [2, 50]),
              sherpa.Choice('criterion', ['gini', 'entropy']),
              sherpa.Continuous('max_features', [0.1, 0.9])]

algorithm = sherpa.algorithms.BayesianOptimization(max_num_trials=100,
                                                   num_grid_points=2)

X, y = load_breast_cancer(return_X_y=True)
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False)

for trial in study:
    clf = RandomForestClassifier(criterion=trial.parameters['criterion'],
                                 max_features=trial.parameters['max_features'],
                                 n_estimators=trial.parameters['n_estimators'],
                                 random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    study.add_observation(trial, iteration=1, objective=scores.mean())
    study.finalize(trial)

time.sleep(60)