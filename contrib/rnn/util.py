from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.hill.ant_hill_env import AntEnv, AntHillEnv
from rllab.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahEnv, HalfCheetahHillEnv
from rllab.envs.mujoco.hill.hopper_hill_env import HopperEnv, HopperHillEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hill.swimmer3d_hill_env import Swimmer3DEnv, Swimmer3DHillEnv
from rllab.envs.mujoco.hill.walker2d_hill_env import Walker2DEnv, Walker2DHillEnv

import contrib.rnn.envs.occluded_envs as occluded_envs

full_constructors = {'swimmer-hill':Swimmer3DHillEnv,
               'ant-hill': AntHillEnv,
               'halfcheetah-hill': HalfCheetahHillEnv,
               'hopper-hill':HopperHillEnv,
               'walker2d-hill':Walker2DHillEnv,
               'swimmer':Swimmer3DEnv,
               'ant': AntEnv,
               'halfcheetah': HalfCheetahEnv,
               'hopper':HopperEnv,
               'walker2d':Walker2DEnv,
               'swimmer2d':SwimmerEnv,
               'swimmerz':Swimmer3DEnv}

def get_base_env(env):
    base_env = env
    done = False
    while not done:
        try:
            base_env = base_env._wrapped_env
        except AttributeError:
            done = True
    return base_env

def construct_env(param_dict):
    env_name = param_dict["env_name"]
    
    if param_dict.get("full_env", True):
        base_env_cls = full_constructors[str.lower(env_name)]
    else:
        base_env_cls = occluded_envs.get_env(env_name)
    base_env = base_env_cls(**param_dict.get("env_kwargs", {}))
    env = normalize(base_env)
    
    real_base_env = get_base_env(base_env)
    env_dt = real_base_env.model.opt.timestep * real_base_env.frame_skip
    return env, env_dt