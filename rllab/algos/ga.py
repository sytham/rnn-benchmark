import numpy as np
from itertools import chain, zip_longest

from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.misc.special import discount_cumsum
from rllab.sampler import parallel_sampler, stateful_pool
import rllab.plotter as plotter
from rllab.sampler.utils import rollout
import rllab.misc.logger as logger

def tournament_select(pop, fitness, thr):
    N = pop.shape[0]
    n = 0
    while n < N:
        candidates = np.random.choice(N, 4, replace=False)
        cfit = fitness[candidates]
        r = np.random.rand(2)
        win_i1 = 0 if cfit[0] > cfit[1] else 1
        lose_i1 = int(not win_i1)
        win_i2 = 0 if cfit[2] > cfit[3] else 1
        lose_i2 = int(not win_i2)
        i1 = candidates[win_i1] if r[0] < thr else candidates[lose_i1]
        i2 = candidates[win_i2+2] if r[1] < thr else candidates[lose_i2+2]
        n += 2
        yield i1, i2

def uniform_crossover(parent1, parent2, pc):
    c = np.random.rand(len(parent1)) < pc
    temp = parent1.copy()
    parent1[c] = parent2[c]
    parent2[c] = temp[c]
    return [parent1, parent2]
    
def mutate(pop, pm, std_multiplier=1.0):
    delta = np.random.standard_normal(pop.shape) * np.std(pop,0) * std_multiplier
    mut = np.random.rand(*pop.shape) < pm
    pop[mut] += delta[mut]

def _get_stderr_lb(x):
    mu = np.mean(x, 0)
    stderr = np.std(x, axis=0, ddof=1 if len(x) > 1 else 0) / np.sqrt(len(x))
    return mu - stderr

def _get_stderr_lb_varyinglens(x):
    mus, stds, ns = [], [], []
    for temp_list in zip_longest(*x, fillvalue=np.nan):
        mus.append(np.nanmean(temp_list))
        n = len(temp_list) - np.sum(np.isnan(temp_list))
        stds.append(np.nanstd(temp_list, ddof=1 if n > 1 else 0))
        ns.append(n)
    return np.array(mus) - np.array(stds) / np.sqrt(ns)
   
def sample_return(G, params, max_path_length, discount, n_evals):
    # env, policy, params, max_path_length, discount = args
    # of course we make the strong assumption that there is no race condition
    G.policy.set_param_values(params)
    paths, returns, undiscounted_returns = [], [], []
    for _ in range(n_evals):
        path = rollout(G.env, G.policy, max_path_length)
        path["returns"] = discount_cumsum(path["rewards"], discount)
        path["undiscounted_return"] = sum(path["rewards"])
        paths.append(path)
        returns.append(path["returns"])
        undiscounted_returns.append(path["undiscounted_return"])
    
    result_path = {'full_paths':paths}
    result_path['undiscounted_return'] = _get_stderr_lb(undiscounted_returns)
    result_path['returns'] = _get_stderr_lb_varyinglens(returns)
    
    return result_path


class GA(RLAlgorithm, Serializable):

    def __init__(
            self,
            env,
            policy,
            n_itr=500,
            max_path_length=500,
            discount=1.0,
            n_samples=100,
            p_crossover=0.7,
            p_mutation=0.1,
            g_mutation = 0.5,
            selection_thr=0.75,
            n_elite=1,
            n_evals=1,
            plot=False,
            **kwargs
    ):
        """
        :param n_itr: Number of iterations.
        :param max_path_length: Maximum length of a single rollout.
        :return:
        """
        Serializable.quick_init(self, locals())
        self.env = env
        self.policy = policy
        self.plot = plot
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_itr = n_itr
        self.n_samples = n_samples
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.n_elite = n_elite
        self.n_evals = n_evals
        self.selection_thr = selection_thr
        self.g_mutation = g_mutation
    
    def train(self):
        parallel_sampler.populate_task(self.env, self.policy)
        if self.plot:
            plotter.init_plot(self.env, self.policy)

        cur_mean = self.policy.get_param_values()
        pop = np.random.standard_normal((self.n_samples, len(cur_mean))) + cur_mean
        last_elite_ix = None
        
        for itr in range(self.n_itr):
            # fitness eval
            infos = (
                stateful_pool.singleton_pool.run_map(sample_return, [(x, self.max_path_length,
                                                                      self.discount, self.n_evals) for x in pop]))
            fs = np.array([info['returns'][0] for info in infos])
            if last_elite_ix is not None:
                print("Elitist return after new eval:", fs[last_elite_ix], "params:", pop[last_elite_ix])
                
            # save elite inds
            elite_ix = (-fs).argsort()[:self.n_elite]
            elites = pop[elite_ix]
            
            # selection, crossover, mutation
            nextpop = []
            for ix1, ix2 in tournament_select(pop, fs, self.selection_thr):
                nextpop.extend(uniform_crossover(pop[ix1,:].copy(), pop[ix2,:].copy(), self.p_crossover))
            nextpop = np.array(nextpop)
            mutate(nextpop, self.p_mutation, self.g_mutation*(1.0 - float(itr)/self.n_itr))
            
            # copy elites
            ix = np.random.choice(nextpop.shape[0], self.n_elite)
            nextpop[ix,:] = elites.copy()
            last_elite_ix = ix[0]
            
            pop = nextpop
            
            print("Elitist return:", fs[elite_ix[0]], "params:", elites[0])
            logger.push_prefix('itr #%d | ' % itr)
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CurStdMean', np.mean(np.std(pop,0)))
            undiscounted_returns = np.array(
                [info['undiscounted_return'] for info in infos])
            logger.record_tabular('AverageReturn',
                                  np.mean(undiscounted_returns))
            logger.record_tabular('StdReturn',
                                  np.std(undiscounted_returns))
            logger.record_tabular('MaxReturn',
                                  np.max(undiscounted_returns))
            logger.record_tabular('MinReturn',
                                  np.min(undiscounted_returns))
            logger.record_tabular('AverageDiscountedReturn',
                                  np.mean(fs))
            logger.record_tabular('NumTrajs',
                                  len(infos))
            infos = list(chain(*[d['full_paths'] for d in infos])) #flatten paths for the case n_evals > 1
            logger.record_tabular('AvgTrajLen',
                                  np.mean([len(path['returns']) for path in infos]))
            
            self.policy.set_param_values(elites[0])
            self.env.log_diagnostics(infos)
            self.policy.log_diagnostics(infos)
            
            logger.save_itr_params(itr, dict(
                itr=itr,
                policy=self.policy,
                env=self.env,
            ))
            logger.dump_tabular(with_prefix=False)
            if self.plot:
                plotter.update_plot(self.policy, self.max_path_length)
            logger.pop_prefix()
        
        # Set final params.
        self.policy.set_param_values(elites[0])
        parallel_sampler.terminate_task()
            

            

                