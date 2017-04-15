from __future__ import absolute_import
from .schedulers import JobScheduler
from .resultstable import ResultsTable
from .hparam_generators import RandomGenerator
from . import Repository
import math
import os


class RandomSearch(object):
    """
    Base class for specific algorithms e.g. Hyper-band

    # Example:

    Hyperband(my_model, my_dataset, my_hparam_ranges, my_scheduler, repo_dir='./example/')

    Initializes entire pipeline

    init hparam_gen(hparam_ranges)
    init results(repo_dir)
    init repo(results, model_func, dataset, repo_dir)
    scheduler.set_repository(repo)
    """
    def __init__(self, model_function, hparam_ranges, repo_dir='./hyperband_repository', dataset=None, **generator_args):
        assert hparam_ranges
        assert model_function
        assert dataset or generator_args, "You need to pass either a dataset array or generator arguments"
        assert set(generator_args.keys()) == {'generator', 'steps_per_epoch', 'validation_data', 'validation_steps'}\
               or set(generator_args.keys()) == {'generator_function', 'steps_per_epoch', 'validation_steps',
                                                 'train_gen_args', 'valid_gen_args'}

        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)

        self.results_table = ResultsTable(repo_dir)

        repo = Repository(model_function=model_function,
                          dataset=dataset,
                          results_table=self.results_table,
                          dir=repo_dir,
                          **generator_args)

        self.scheduler = JobScheduler(repository=repo)
        # Note, if we pass a scheduler we still need to pass the repo to it

        self.hparam_gen = RandomGenerator(hparam_ranges)

    def run(self, num_experiments, num_epochs):
        """
        Args:
            num_experiments:
            num_epochs:

        Returns:

        """
        run = 1
        for id in range(num_experiments):
            self.scheduler.submit(run_id=(run, id), hparams=self.hparam_gen.next(), epochs=num_epochs)
            print(self.results_table._get_table())

        return self.results_table._get_table()


class Hyperband(RandomSearch):
    """
    Child class for Hyper-band algorithm
    """
    def __init__(self, **kwargs):
        super(Hyperband, self).__init__(**kwargs)

    def run(self, R=20, eta=3):
        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))  # initial number of configurations
            r = R * eta ** (-s)  # initial number of iterations to run configurations for

            for i in range(s + 1):
                # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                if i == 0:
                    [self.scheduler.submit(run_id=(s, j), hparams=self.hparam_gen.next(), epochs=r_i) for j in range(n_i)]
                else:
                    [self.scheduler.submit(run_id=(s, T_j), epochs=r_i) for T_j in self.results_table.get_k_lowest_from_run(n_i, s)]

        return self.results_table._get_table()
