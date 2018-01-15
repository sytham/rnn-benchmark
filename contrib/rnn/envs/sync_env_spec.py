from rllab.core.serializable import Serializable
from rllab.envs.env_spec import EnvSpec

class SyncEnvSpec(EnvSpec):
    def __init__(self,
                 observation_space,
                 action_space,
                 joint_sync_offsets):
        '''
        :param joint_sync_offsets: <num_motors x num_motors> ndarray
            A zero entry at i,j specifies no edge between joint i and j. A non-zero entry specifies joint j should sync
            with joint i, with the phase offset given by the entry. Edges can pretty much correspond to robot morphology,
            e.g. knee joint syncs to hip joint etc.
        '''
        Serializable.quick_init(self, locals())
        super(SyncEnvSpec, self).__init__(observation_space, action_space)
        self._joint_sync_offets = joint_sync_offsets
    
    @property
    def joint_sync_offsets(self):
        return self._joint_sync_offets
        