import numpy as np
from rllab.envs.occlusion_env import OcclusionEnv
from rllab.core.serializable import Serializable
from rllab.envs.mujoco.hill.ant_hill_env import AntEnv, AntHillEnv
from rllab.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahEnv, HalfCheetahHillEnv
from rllab.envs.mujoco.hill.hopper_hill_env import HopperEnv, HopperHillEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.hill.swimmer3d_hill_env import Swimmer3DEnv, Swimmer3DHillEnv
from rllab.envs.mujoco.hill.walker2d_hill_env import Walker2DEnv, Walker2DHillEnv
from contrib.rnn.envs.sync_env_spec import SyncEnvSpec
from cached_property import cached_property

''' ===== Sync specs for oscillator controller init (only Rossler atm) ==== '''
def halfcheetah_sync_spec():
    K = np.zeros((6,6)) # 6 motorized joints. Hind hip is "master" that others sync to
    K[0,1] = np.pi # hind knee to hind hip
    K[1,2] = np.pi # hind ankle to hind knee
    K[0,3] = 0.8*np.pi # front hip to hind hip
    K[3,4] = 0.33*np.pi # front knee to front hip
    K[3,5] = np.pi # front foot to front hip
    return K

def hopper_sync_spec():
    K = np.zeros((3,3)) # 3 motorized joints. Leg joint is "master" that others sync to
    K[1,2] = 0.2*np.pi # foot to leg
    K[1,0] = np.pi # thigh to leg
    return K

def swimmer_sync_spec():
    # traveling wave
    return np.array([[0.0, -0.2*np.pi], [0.0, 0.0]])
      
''' ============= REGULAR ENVS ============== '''
class OccludedSwimmerEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmerEnv, self).__init__(SwimmerEnv(), [2,3,4]) # joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            joint_sync_offsets=swimmer_sync_spec()
        )
        
class OccludedSwimmer3DEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmer3DEnv, self).__init__(Swimmer3DEnv(), [6,7,8]) # yaw and joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            joint_sync_offsets=swimmer_sync_spec()
        )
        
class OccludedSwimmer3DZEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedSwimmer3DZEnv, self).__init__(Swimmer3DEnv(), [2,5,6,7,8]) #z-pos, pitch, yaw, joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            joint_sync_offsets=swimmer_sync_spec()
        )
        
class OccludedAntEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedAntEnv, self).__init__(AntEnv(), list(range(2,15)) + [57,75,93,111]) # joint angles, leg ground normal contact force
        
class OccludedHalfCheetahEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedHalfCheetahEnv, self).__init__(HalfCheetahEnv(), list(range(8))) # z-pos, pitch, joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
        observation_space=self.observation_space,
        action_space=self.action_space,
        joint_sync_offsets=halfcheetah_sync_spec()
    )

class OldOccludedHopperEnv(OcclusionEnv):
    def __init__(self):
        super(OldOccludedHopperEnv, self).__init__(HopperEnv(alive_coeff=0.1), [0,1,2,3,4,11,12,13,14,15,16]) # z-pos, joint angles, constraint forces
          
class OccludedHopperEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedHopperEnv, self).__init__(HopperEnv(alive_coeff=0.1), range(17)) #[0,1,2,3,4,11,12,13,14,15,16]) # z-pos, joint angles, constraint forces
        
class OccludedWalker2DEnv(OcclusionEnv):
    def __init__(self):
        super(OccludedWalker2DEnv, self).__init__(Walker2DEnv(), list(range(2,10))) # joint angles (why no constraint forces here?)


''' ============= HILL ENVS ============== '''
class OccludedSwimmer3DHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedSwimmer3DHillEnv, self).__init__(Swimmer3DHillEnv(*args, **kwargs), [2,5,6,7,8]) #z-pos, pitch, yaw, joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
            joint_sync_offsets=swimmer_sync_spec()
        )
         
class OccludedAntHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedAntHillEnv, self).__init__(AntHillEnv(*args, **kwargs), list(range(2,15)) + [57,75,93,111]) # joint angles, leg ground normal contact force
        
class OccludedHalfCheetahHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedHalfCheetahHillEnv, self).__init__(HalfCheetahHillEnv(*args, **kwargs), list(range(8))) # z-pos, pitch, joint angles
    @cached_property
    def spec(self):
        return SyncEnvSpec(
        observation_space=self.observation_space,
        action_space=self.action_space,
        joint_sync_offsets=halfcheetah_sync_spec()
    )

class OldOccludedHopperHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OldOccludedHopperHillEnv, self).__init__(HopperHillEnv(*args, alive_coeff=0.1, **kwargs), [0,1,2,3,4,11,12,13,14,15,16]) #joint angles, constraint forces
                
class OccludedHopperHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedHopperHillEnv, self).__init__(HopperHillEnv(*args, alive_coeff=0.1, **kwargs), range(17)) #  [0,1,2,3,4,11,12,13,14,15,16] joint angles, constraint forces
        
class OccludedWalker2DHillEnv(OcclusionEnv):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super(OccludedWalker2DHillEnv, self).__init__(Walker2DHillEnv(*args, **kwargs), list(range(2,10))) # joint angles (why no constraint forces here?)

constructors = {'swimmer-hill':OccludedSwimmer3DHillEnv,
               'ant-hill': OccludedAntHillEnv,
               'halfcheetah-hill': OccludedHalfCheetahHillEnv,
               'hopper-hill':OccludedHopperHillEnv,
               'walker2d-hill':OccludedWalker2DHillEnv,
               'swimmer':OccludedSwimmer3DEnv,
               'ant': OccludedAntEnv,
               'halfcheetah': OccludedHalfCheetahEnv,
               'hopper':OccludedHopperEnv,
               'walker2d':OccludedWalker2DEnv,
               'swimmer2d':OccludedSwimmerEnv,
               'swimmerz':OccludedSwimmer3DZEnv}

def get_env(name):
    return constructors[str.lower(name)]

def list_envs():
    return constructors.keys()
    