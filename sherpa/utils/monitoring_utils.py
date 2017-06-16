from __future__ import print_function
from __future__ import division
import math
import sys
import time
import re
import numpy as np
from subprocess import check_output


def timedcall(fn, kwargs):
    "Call function with args; return the time in seconds and result."
    t0 = time.clock()
    result = fn(**kwargs)
    t1 = time.clock()
    return t1-t0, result


def visualize_hyperband_params(R=None, eta=None):
    """
    This function visualizes the training schedule for any values of R and
    eta in a table format similar to that in [Kevin Jamieson's Blog post](
    https://people.eecs.berkeley.edu/~kjamieson/hyperband.html).

    # Arguments
    R: The maximum number of epochs per 'stage', also max_iter in the blog post
    eta: The cut-factor.

    """
    if not R:
        if sys.version.startswith('2'):
            R = raw_input("Enter R: ") or '81'
        else:
            R = input("Enter R (default 81): ") or '81'
    if not eta:
        if sys.version.startswith('2'):
            eta = raw_input("Enter eta (default 3): ") or '3'
        else:
            eta = input("Enter eta (default 3): ") or '3'
    R = float(R)
    eta = float(eta)

    log_eta = lambda x: math.log(x) / math.log(eta)
    s_max = int(log_eta(R))
    B = (s_max + 1) * R

    run_line = []
    header_line = ['']
    stages = [['Cont'] for i in range(s_max+1)]
    stages[0][0] = 'Init'
    total_epochs = 0

    for s in reversed(range(s_max + 1)):
        n = int(math.ceil(B / R / (s + 1) * eta ** s))
        r = R * eta ** (-s)

        run_line.append('run={}'.format(s_max - s + 1))
        header_line.append('models\tepochs')

        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = int(round(r * eta ** (i)))

            total_epochs += n_i * r_i

            stages[i].append('{}\t{}'.format(n_i, r_i))
    print('\n\n')
    print('-' * 100)
    print('\t' + '\t\t'.join(run_line))
    print('\t'.join(header_line))
    for stage in stages:
        print('\t'.join(stage))

    print('-'*100)
    print('Total epochs={}'.format(total_epochs))
    print('-' * 100)
    print('\n\n')

    return total_epochs


def extract_loss_history_from_log(path, key='val_loss', lower_is_better=True):
    losses = []
    best_loss = np.inf if lower_is_better else -np.inf
    with open(path, 'rb') as f:
        for line in f:
            this_loss = re.findall(r'{}: (\d\.[0-9]+)'.format(key),
                                   line.decode('utf-8'))
            if this_loss == []:
                continue
            this_loss = float(this_loss[0])
            best_loss = min(this_loss, best_loss) if lower_is_better else \
                max(this_loss, best_loss)
            losses.append(best_loss)
    return losses


def read_nvidia_smi(gpus=list(range(4)), cutoff=60):
    """
    Args:
        gpus: GPUs to be checked
        cutoff: Usage (in MiB) above which a gpu is declared to be in use

    Returns:
        True/False depending on whether those gpus are in use
    """

    s = check_output(["nvidia-smi"])
    matches = re.findall(r'([0-9]+)MiB / [0-9]+MiB', s)
    usage = map(int, matches)
    return all(gpu_usage >= cutoff for gpu_id, gpu_usage in enumerate(usage) if gpu_id in gpus)


def gpu_exists():
    """
    Returns:
        True/False depending on whether there is at least one GPU installed
    """
    try:
        check_output(["nvidia-smi"])
        return True
    except(OSError):
        return False


if __name__ == '__main__':
    visualize_hyperband_params()