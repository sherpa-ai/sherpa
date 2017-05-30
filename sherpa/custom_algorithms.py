from __future__ import absolute_import
from __future__ import division
from .hparam_generators import RandomGenerator, GaussianProcessEI, LatinHypercube
from .utils.monitoring_utils import visualize_hyperband_params, timedcall
from .algorithms import Algorithm
import math


class BayesianOptimization(Algorithm):
    def __init__(self, model_function, hparam_ranges,
                 repo_dir='./hyperband_repository', loss='val_loss',
                 dataset=None,
                 generator_function=None, train_gen_args=None,
                 steps_per_epoch=None, validation_data=None,
                 valid_gen_args=None, validation_steps=None):
        super(self.__class__, self).__init__(model_function=model_function,
                                             loss=loss,
                                        repo_dir=repo_dir,
                                        dataset=dataset,
                                        generator_function=generator_function,
                                        train_gen_args=train_gen_args,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        valid_gen_args=valid_gen_args,
                                        validation_steps=validation_steps)
        self.hparam_gen = GaussianProcessEI(hparam_ranges)

    def run(self, num_experiments, num_epochs):
        run = 1
        for id in range(num_experiments):
            X = self.results_table.get_hparams_df(as_design_matrix=True)
            y = self.results_table.get_column('Loss')
            next_hparams = self.hparam_gen.next(X=X, y=y)
            self.scheduler.submit(run_id='{}_{}'.format(run, id),
                                  hparams=next_hparams,
                                  epochs=num_epochs)
            # print(self.results_table.get_table())

        return self.results_table.get_table()





class TemperatureHyperband(Algorithm):
    def __init__(self, model_function, hparam_ranges,
                 repo_dir='./hyperband_repository', loss='val_loss',
                 dataset=None,
                 generator_function=None, train_gen_args=None,
                 steps_per_epoch=None, validation_data=None,
                 valid_gen_args=None, validation_steps=None):
        super(self.__class__, self).__init__(model_function=model_function,
                                             loss=loss,
                                        repo_dir=repo_dir,
                                        dataset=dataset,
                                        generator_function=generator_function,
                                        train_gen_args=train_gen_args,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        valid_gen_args=valid_gen_args,
                                        validation_steps=validation_steps)
        self.hparam_gen = RandomGenerator(hparam_ranges)

    def run(self, R=20, eta=3, temperature=1.):
        total_epochs = visualize_hyperband_params(R=R, eta=eta)

        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))
            r = R * eta ** (-s)

            for i in range(s + 1):
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                run = s_max - s + 1
                if i == 0:
                    for j in range(1, n_i + 1):
                        if s == s_max and i == 0 and j == 1:
                            self.estimate_time(self.scheduler.submit,
                                               {'run_id': '{}_{}'.format(run,j),
                                                'hparams':
                                                    self.hparam_gen.next(),
                                                'epochs': r_i},
                                               total_epochs=total_epochs,
                                               r_i=r_i)

                        else:
                            self.scheduler.submit(run_id='{}_{}'.format(run,
                                                                        j),
                                                  hparams=self.hparam_gen.next(),
                                                  epochs=r_i)
                else:
                    for run_id in self.results_table.sample_k_ids_from_run(n_i, run):
                        self.scheduler.submit(run_id=run_id, epochs=r_i)

        return self.results_table.get_table()

    @staticmethod
    def estimate_time(f, args, total_epochs, r_i):
        time, result = timedcall(f, args)

        secs = total_epochs * time / r_i
        hrs = secs // 3600
        mins = (secs % 3600) // 60
        print('-' * 100)
        print('\nThe complete Hyperband optimization is '
              'estimated to take {}hrs and {} '
              'mins\n'.format(
            hrs, mins))
        print('-' * 100)




class Hyperbayes(Algorithm):
    def __init__(self, model_function, hparam_ranges,
                 repo_dir='./hyperbayes_repository', loss='val_loss',
                 dataset=None,
                 generator_function=None, train_gen_args=None,
                 steps_per_epoch=None, validation_data=None,
                 valid_gen_args=None, validation_steps=None):
        super(self.__class__, self).__init__(model_function=model_function,
                                             loss=loss,
                                        repo_dir=repo_dir,
                                        dataset=dataset,
                                        generator_function=generator_function,
                                        train_gen_args=train_gen_args,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        valid_gen_args=valid_gen_args,
                                        validation_steps=validation_steps)
        self.hparam_gen = GaussianProcessEI(hparam_ranges)

    def run(self, R=20, eta=3):

        total_epochs = visualize_hyperband_params(R=R, eta=eta)

        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))
            r = R * eta ** (-s)

            for i in range(s + 1):
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                run = s_max - s + 1
                if i == 0:
                    for j in range(1, n_i+1):
                        X = self.results_table.get_hparams_df(
                            as_design_matrix=True)
                        y = self.results_table.get_column('Loss')
                        next_hparams = self.hparam_gen.next(X=X, y=y)
                        self.scheduler.submit(run_id='{}_{}'.format(run,j),
                                                  hparams=next_hparams,
                                                  epochs=r_i)
                else:
                    for run_id in self.results_table.get_k_lowest_from_run(n_i,
                                                                        run):
                        self.scheduler.submit(run_id=run_id, epochs=r_i)

        return self.results_table.get_table()

class Legoband(Algorithm):
    def __init__(self, model_function, hparam_ranges,
                 repo_dir='./hyperbayes_repository', loss='val_loss',
                 dataset=None,
                 generator_function=None, train_gen_args=None,
                 steps_per_epoch=None, validation_data=None,
                 valid_gen_args=None, validation_steps=None):
        super(self.__class__, self).__init__(model_function=model_function,
                                             loss=loss,
                                        repo_dir=repo_dir,
                                        dataset=dataset,
                                        generator_function=generator_function,
                                        train_gen_args=train_gen_args,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        valid_gen_args=valid_gen_args,
                                        validation_steps=validation_steps)
        self.hparam_gen = RandomGenerator(hparam_ranges)

    def run(self, R=20, eta=3):

        total_epochs = visualize_hyperband_params(R=R, eta=eta)

        log_eta = lambda x: math.log(x) / math.log(eta)
        s_max = int(log_eta(R))
        B = (s_max + 1) * R

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / R / (s + 1) * eta ** s))
            r = R * eta ** (-s)

            for i in range(s + 1):
                n_i = int(n * eta ** (-i))
                r_i = int(round(r * eta ** (i)))

                run = s_max - s + 1
                if i == 0:
                    for j in range(1, n_i+1):
                        self.scheduler.submit(run_id='{}_{}'.format(run,j),
                                              hparams=self.hparam_gen.next(),
                                              epochs=r_i)
                        self.grow_distributions(run_id='{}_{}'.format(run,j),
                                                run=run,
                                                epochs=r_i)


                else:
                    for T_j in self.results_table.get_k_lowest_from_run(n_i,
                                                                        run):
                        self.scheduler.submit(run_id=(run, T_j), epochs=r_i)
                        self.grow_distributions(run_id='{}_{}'.format(run,j),
                                                run=run,
                                                epochs=r_i)

        return self.results_table.get_table()

    def grow_distributions(self, run_id, run, epochs):
        best_id = self.results_table.get_k_lowest_from_run(1, run=run)[0]
        amount = epochs if best_id == run_id else -epochs
        hparams = self.results_table.get(run_id=run_id,
                                         parameter='Hparams')
        hparams = eval(hparams)
        self.hparam_gen.grow(hparams, amount)

class NaturalSelection(Algorithm):
    def __init__(self, model_function, hparam_ranges,
                 repo_dir='./natural_selection_repository', loss='val_loss',
                 dataset=None,
                 generator_function=None, train_gen_args=None,
                 steps_per_epoch=None, validation_data=None,
                 valid_gen_args=None, validation_steps=None):
        super(self.__class__, self).__init__(model_function=model_function,
                                             loss=loss,
                                        repo_dir=repo_dir,
                                        dataset=dataset,
                                        generator_function=generator_function,
                                        train_gen_args=train_gen_args,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=validation_data,
                                        valid_gen_args=valid_gen_args,
                                        validation_steps=validation_steps)
        self.hparam_gen = LatinHypercube(hparam_ranges)

    def run(self, factor=6):
        id = 1
        for run in range(factor):
            n_i = 2**(factor-1)/(2**run)
            r_i = 2**run
            k = 0 if run==0 else n_i//2

            for run_id in self.results_table.get_k_lowest_from_run(int(k),
                                                                   run-1):
                self.scheduler.submit(run_id=run_id, epochs=r_i)
                self.grow_dist(run_id=run_id, epochs=r_i/2)
                self.results_table.set_value(run_id=run_id, col='Run',
                                             value=run)

            for i in range(int(n_i - k)):
                self.scheduler.submit(run_id='{}_{}'.format(run, id),
                                      epochs=r_i,
                                      hparams=self.hparam_gen.next())
                id += 1

        return self.results_table.get_table()

    def grow_dist(self, run_id, epochs):
        hparams = self.results_table.get(run_id=run_id,
                                         parameter='Hparams')
        hparams = eval(hparams)
        self.hparam_gen.grow(hparams=hparams, amount=epochs)
