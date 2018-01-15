import numpy as np

from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.sampler.base import BaseSampler
from rllab.sampler import parallel_sampler, stateful_pool
from rllab.misc.special import discount_cumsum
from rllab.sampler.utils import rollout

def get_base_env(env):
    base_env = env
    done = False
    while not done:
        try:
            base_env = base_env._wrapped_env
        except AttributeError:
            done = True
    return base_env

def sample_return(G, params, max_path_length, discount, env_state, agent_state=None):
    # env, policy, params, max_path_length, discount = args
    # of course we make the strong assumption that there is no race condition
    G.policy.set_param_values(params)
    path = rollout(G.env, G.policy, max_path_length, env_start_state=env_state, agent_start_state=agent_state)
    path["returns"] = discount_cumsum(path["rewards"], discount)
    path["undiscounted_return"] = sum(path["rewards"])
    return path

class ScenarioSampler(BaseSampler):
    def __init__(self, algo, forward=True):
        self.algo = algo
        self.forward = forward
        self.base_env = get_base_env(self.algo.env)
        
    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        self.algo.env.reset()
        env_state = self.base_env._full_state
        agent_state = None
        if self.algo.policy.recurrent:
            self.algo.policy.reset()
            agent_state = self.algo.policy.get_state()
        
        theta = self.algo.policy.get_param_values()
        eps = 1e-7 / (np.linalg.norm(theta) + 1e-8)
        scenarios = np.random.standard_normal((self.algo.batch_size+1, len(theta)))*eps + theta
        scenarios[0,:] = theta[:]
                        
        paths = (stateful_pool.singleton_pool.run_map(
            sample_return,
                [(x, self.algo.max_path_length,
                self.algo.discount, env_state, agent_state) for x in scenarios]
                ))
        
        return paths

        
class FiniteDifferences(BatchPolopt):
    def __init__(self,
                 sampler_cls=ScenarioSampler,
                 **kwargs):
        super(FiniteDifferences, self).__init__(sampler_cls=sampler_cls, **kwargs)
    
    @overrides
    def init_opt(self):
        pass
    
    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
        
    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        #dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        #all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
#         loss_before = self.optimizer.loss(all_input_values)
#         mean_kl_before = self.optimizer.constraint_val(all_input_values)
#         self.optimizer.optimize(all_input_values)
#         mean_kl = self.optimizer.constraint_val(all_input_values)
#         loss_after = self.optimizer.loss(all_input_values)
#         logger.record_tabular('LossBefore', loss_before)
#         logger.record_tabular('LossAfter', loss_after)
#         logger.record_tabular('MeanKLBefore', mean_kl_before)
#         logger.record_tabular('MeanKL', mean_kl)
#         logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()
    