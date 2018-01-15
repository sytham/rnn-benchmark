import numpy as np
from rllab.misc import tensor_utils
import time
import warnings

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, env_start_state=None, agent_start_state=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    if env_start_state is not None: # not all envs support init_state
        o = env.reset(init_state=env_start_state)
    else:
        o = env.reset()
    if agent_start_state is not None:
        agent.reset(init_state=agent_start_state)
    else:
        agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if np.any(np.logical_or(np.logical_or(np.isinf(a), np.isnan(a)), np.abs(a) > 1e3)):
            warnings.warn("Invalid action detected")
            rewards[-1] = -1000.0
            break
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
