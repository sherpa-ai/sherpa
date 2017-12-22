from __future__ import print_function
import sherpa


parameters = [sherpa.Choice(name="param_a",
                            range=[1, 2, 3]),
              sherpa.Continuous(name="param_b",
                                range=[0, 1])]

algorithm = sherpa.RandomSearch(max_num_trials=50)
stopping_rule = sherpa.MedianStoppingRule(min_iterations=2,
                                          min_trials=5)
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     stopping_rule=stopping_rule,
                     lower_is_better=True)

num_iterations = 10

for trial in study:
    print("Trial {}:\t{}".format(trial.id, trial.parameters))

    # Simulate model training
    for i in range(num_iterations):
        pseudo_objective = 1/float(i+1) * trial.parameters['param_b']
        study.add_observation(trial=trial,
                              iteration=i+1,
                              objective=pseudo_objective)
        if study.should_trial_stop(trial=trial):
            print("Stopping Trial {} after {} iterations.".format(trial.id, i+1))
            break
            
    study.finalize(trial=trial,
                   status='COMPLETED')

print(study.results)