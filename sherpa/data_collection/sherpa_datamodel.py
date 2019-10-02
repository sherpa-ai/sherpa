from rtypes import pcc_set, merge
from rtypes import dimension, primarykey
import numpy as np

'''
    Each result dictionary should contain result_id - result value pairs, result value is
    in the form of (objective, iteration, context)
'''

# pcc_set client_id -- Client_set
@pcc_set
class Client_set(object):
    client_id = primarykey(int)
    name = dimension(str)
    result = dimension(list)
    ready_for_new_trial_result = dimension(bool)
    assigned_trial_result = dimension(int)
    '''
    Args:
            client_id (int): unique id for each client
            name (str): name for each client
            result (list): dictionary that stores all values for results in Trial_Results sets

            trial (sherpa.core.Trial): trial to send metrics for.
            iteration (int): the iteration e.g. epoch the metrics are for.
            objective (float): the objective value.
            context (dict): other metric-values.
    '''
    def __init__(self, client_id, name, result):
        self.client_id = client_id
        self.name = name
        self.result = result
        self.ready_for_new_trial_result = False # initialized as False
        self.assigned_trial_result = -1 # initialized as -1


# pcc_set trial_id -- Trial_Results
@pcc_set
class Trial_Results(object):
    trial_id = primarykey(int)
    parameters = dimension(dict)
    meta = dimension(dict)
    assigned_client = dimension(int)
    completed = dimension(bool)
    name = dimension(str)
    result = dimension(list)

    '''
    Args:
            tiral (sherpa.core.Trial): trial to send metrics for.
            name (str): name for each trial

    '''
    def __init__(self, trial, name):
        self.trial_id = trial.id
        self.parameters = trial.parameters
        self.meta = {}
        self.assigned_client = -1
        self.completed = False
        self.name = name
        self.result = [] # dict becomes None, but int/float/list are fine
