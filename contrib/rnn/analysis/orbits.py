import sys
from collections import defaultdict, OrderedDict
from importlib import import_module
import argparse
import joblib
import numpy as np
import pickle, h5py

from contrib.rnn.util import get_base_env

def _copy_mujoco_state(env):
    m = env.model.data
    d = OrderedDict()
    d['time'] = m.time
    d['qpos'] = m.qpos.copy()
    d['qvel'] = m.qvel.copy()
    d['qacc'] = m.qacc.copy()
    return d

def _print_mujoco_state(d):
    for k,v in d.items():
        print(k, v)
    
def _set_mujoco_state(env, d):
    m = env.model.data
    m.time = d['time']
    m.qpos = d['qpos'].copy()
    m.qvel = d['qvel'].copy()
    m.qacc = d['qacc'].copy()

def _get_state(env, agent):
    #state = env.get_full_state()
    #state['uncontrolled_env_state'] = env.get_uncontrolled_state()
    state = _copy_mujoco_state(env)
    state['uncontrolled_env_state'] = {}
    state['agent_state'] = agent.get_state()
    return state

def _set_state(env, agent, step, state):
    #env.set_uncontrolled_state(state['uncontrolled_env_state'])
    # only set complete system state at the first step. Thereafter either let it diverge or
    # compare system evolves exactly according to previously recorded states for debugging
    if step == 0:
        #env.set_full_state(state)
        _set_mujoco_state(env, state)
        agent.set_state(state['agent_state'])

def strip_dict(d, keys_to_strip):
    newd = OrderedDict()
    for k, v in d.items():
        if k in keys_to_strip:
            continue
        if isinstance(v, dict):
            newd[k] = strip_dict(v, keys_to_strip)
        else:
            newd[k] = v
    return newd

def strip_orbit(o, keys_to_strip):
    return [strip_dict(d, keys_to_strip) for d in o]

def get_entry(d, key):
    keys = key.split(':')
    for k in keys:
        d = d[k]
    return d
        
def ravel_dict(d):
    '''returns list of arrays'''
    s = []
    names = []
    for k, v in d.items():
        if k == 'qacc':
            continue
        if isinstance(v, dict):
            s.extend(ravel_dict(v))
        else:
            s.append(np.atleast_1d(np.array(v, dtype=np.float64).ravel()))
            names.append(k)
    return s

def ravel_dict_names(d):
    '''returns list of names'''
    s = []
    for k, v in d.items():
        if k == 'qacc':
            continue
        if isinstance(v, dict):
            s.extend([k + ':' + e for e in ravel_dict_names(v)])
        else:
            s.append(k)
    return s
        
def _save_data(data, ob, action, rew, agentinfo, envinfo):
    data["observation"].append(ob)            
    data["action"].append(action)
    for (k,v) in agentinfo.items():
        data[k].append(v)
    data["reward"].append(rew)
    for (k,v) in envinfo.items():
        data[k].append(v)

def rollout(env, agent, timestep_limit, record=False, playback=None, regular_data=True, video=False):
    """
    Simulate the env and agent for timestep_limit steps
    """
    base_env = get_base_env(env)
    ob = env.reset()
    agent.reset()
    terminated = False
    data = defaultdict(list)
    record_data = []
    if video:   env.render()
    for i in range(timestep_limit):
        if playback:
            _set_state(base_env, agent, i, playback[i])
            if i == 0:
                ob = env.get_current_obs() 
        if record:
            record_data.append( _get_state(base_env, agent) )
        
        action, agentinfo = agent.get_action(ob)
        ob,rew,done,envinfo = env.step(action)
        if video:   env.render()
        
        if regular_data:
            _save_data(data, ob, action, rew, agentinfo, envinfo)
                
        if done:
            terminated = True
            break
    data = {k:np.array(v) for (k,v) in data.items()}
    data["terminated"] = terminated
    data["full_state_path"] = record_data
    return data

def get_agent_cls(name):
    p, m = name.rsplit('.', 1)
    mod = import_module(p)
    constructor = getattr(mod, m)
    return constructor

def load_agent_from_snapshot(hdf_filename, snapname=None):
    hdf = h5py.File(hdf_filename,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print("snapshots:\n",snapnames)
    if snapname is None: 
        snapname = snapnames[-1]
    elif snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%snapname)
    else: 
        snapname = snapname
    print("Loading snapshot", snapname)
    return pickle.loads(hdf['agent_snapshots'][snapname].value)

def _create_dict_from_options(opt_list):
    d = {}
    for opt in opt_list:
        d[opt[0]] = opt[1](opt[2])
    return d

def _update_dict_from_options(opt_list, d):
    for opt in opt_list:
        d[opt[0]] = opt[1](opt[2])
        
def logdistance_curve(orbits):
    deltas = np.array( [np.linalg.norm(orbits[i].squeeze() - orbits[i+1].squeeze(), axis=1) for i in range(0,len(orbits),2)] )
    return np.mean(np.log(deltas),0)

def _one_standalone_orbit(net, h, x, timestep, N):
    net.h[:] = h.copy()
    for _ in range(N):
        net.output(x, timestep)
        yield net.h.copy()
    
def generate_orbits_standalone_net(net, timestep, with_bias=False, N=10000):
    '''
    Generate orbits for a net on its own, i.e. in the absence of sensory input.
    
    This method samples 10 close (1e-9) random starting points for every 500th point along an initially generated orbit
    from a starting point generated by calling net.reset().
    Thus, for N=1000, the method uses 2 sample points, sampling a cluster of 10 points for each, giving 2*10 orbits.
    
    @param net: object. The network
    @param timestep: float, timestep dt
    @param with_bias: bool, whether to apply input bias weight
    @param N: int, number of steps to simulate
    @return ndarray: array of 10*N/500 x N orbits, computed according to above procedure.
    '''
    if N < 1000:
        raise ValueError("N hsould be larger than 1000")
    x = np.zeros((net.num_in,1))
    if not with_bias:
        net.W = np.zeros_like(net.W)
    net.reset()
    state_dim = net.h.shape[0]
    
    # generate an orbit
    orbit = np.array([h for h in _one_standalone_orbit(net, net.h, x, timestep, N)])
   
    # sample clusters of points along the orbit
    start_states = np.concatenate([orbit[500 + i,:] + 1e-6*np.random.randn(10,state_dim,1) for i in range(0,N-500,500)])
    
    # generate orbits from the sample points
    return [np.array([z for z in _one_standalone_orbit(net, start_state, x, timestep, N)]) for start_state in start_states]

def _one_agent_env_orbit(env, agent, N, record=False, playback=None, regular_data=True):
    return rollout(env, agent, N, record, playback, regular_data)

def _create_perturbed_state(state_dict):
    d = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(v, dict):
            d[k] = _create_perturbed_state(v)
        elif k == 'uncontrolled_env_state': # this, along with non-array entries, should not be perturbed
            d[k] = v
        else:
            try:
                d[k] = np.array(v.copy() + 1e-8*np.random.randn(*v.shape))
            except AttributeError:
                d[k] = v
    return d

def _perturbed_playback_generator(orbit, M):
    '''
    Generate M perturbed starting states, starting from the first state in orbit, for playback.
    '''
    for _ in range(M):
        # strip the path of everything except uncontrolled state -- this should be kept the same for all orbits
        stripped_path = [{'uncontrolled_env_state':d['uncontrolled_env_state']} for d in orbit[1:]]
        stripped_path.insert(0, _create_perturbed_state(orbit[0]))
        yield stripped_path
        
def generate_env_agent_orbits(env, agent, N=10000, cluster_size=10, sample_step_size=500, start=500, exact=False):
    '''
    Generator that generates orbits for a complete environment-agent system.
    
    This method samples <cluster_size> close (1e-8) random starting points for every <sample_step_size> point along an
    initially generated orbit from a randomly chosen starting point by env.reset(). It then generates orbits of length
    N/2 for every one of those starting points.
    Thus, for N=1000, cluster_size=10, sample_step_size=500, the method uses 2 sample points, sampling a cluster of 10
    points for each, giving 2*10 orbits of length 500.
    
    Each orbit is yielded as a list of OrderedDicts; each list entry is the full state at that step, constructed by calling
    get_state() methods of agent and environment.
    @param exact bool, if False, sample points are sampled randomly, and sample_step_size is taken to be avg dist between sample points 
    ''' 
    steps = N/2
        
    # generate an orbit
    orbit = _one_agent_env_orbit(env, agent, N, record=True, regular_data=True)['full_state_path']
    
    if exact:
        sample_points = range(start,N-steps,sample_step_size)
    else:
        npoints = (N - steps - start) / sample_step_size
        sample_points = np.sort(np.random.choice(np.arange(start, N-steps), size=npoints, replace=False))
        
    # sample clusters of starting points along the path
    perturbed_states_for_playback = [o for i in sample_points
                                     for o in _perturbed_playback_generator(orbit[i:i+steps], cluster_size)]
    
    print("Starting playback")
    for o in perturbed_states_for_playback:
        yield _one_agent_env_orbit(env, agent, steps, record=True, playback=o, regular_data=False)['full_state_path']

def array_rep_generator(orbits):
    '''Convert a list of orbits in dict form to a list of arrays'''
    for o in orbits:
        yield np.array([np.concatenate(ravel_dict(s)) for s in o])

def convert_dict_orbits_to_array(orbits):
    return [o for o in array_rep_generator(orbits)]
   
def generate_orbits(env, agent, sensory_input=False, with_bias=False):
    net = agent.policy.net
    if not sensory_input:
        return generate_orbits_standalone_net(net, agent.policy._timestep, with_bias)
    
def generate_orbits_from_snapshot(hdf_filename, snapshot=None, sensory_input=False, with_bias=False):
    if not sensory_input:
        agent = load_agent_from_snapshot(hdf_filename, snapshot)
        return generate_orbits(None, agent, sensory_input=sensory_input, with_bias=with_bias)

def load_env_agent_snapshot(filename):
    data = joblib.load(filename)
    policy = data['policy']
    env = data['env']
    return env, policy


import os
from copy import deepcopy
def get_max_theta(pathexp, *pathargs, **kwargs):
    url = os.path.expandvars(pathexp.format(*pathargs))
    store = h5py.File(url, "r")
    max_scores = store['diagnostics']['ymax'][:,:]
    th_max = store['diagnostics']['th_max'][:,:]
    # flatten th_max except last dim and index into it with argmax of max_scores
    best_theta = th_max.reshape(-1, th_max.shape[-1])[max_scores.argmax()]
    print("Theta for max_score {0:.2f} =".format(max_scores.max()), best_theta)
    return best_theta
  
if __name__ == "__main__":
    net_type = 'discrete_time'
    transferf = 'logmap'
    theta_encoding = 'free'
    env, agent = create_env_and_agent_from_args('modular_rl.agentzoo.DeterministicRNNAgent',
                                                'POFixedTimedViscositySwimmer-v0',
                                                timestep=0.005,
                                                hid_sizes=[10,5],
                                                net_type=net_type,
                                                transfer_function=transferf,
                                               theta_encoding=theta_encoding,
                                               difficulty_level=1.0)
    
    base_url = '$HOME/phd/swimmer_results/cem-rnn-ffwd-results/cem-{0}-{1}-free-timedswimmer-d{2}'
    theta = get_max_theta(base_url, net_type, transferf, '1')
    agent.set_from_flat(theta)
    
    keys_to_strip = ['time', 'iter', 'outputs', 'obfilt_state', 'rewfilt_state']
    orbits = [strip_orbit(o, keys_to_strip) for o in generate_env_agent_orbits(env, agent)]
    
    #orbits = [ o for o in generate_env_agent_orbits(env, agent)]
#     
#     path = _one_agent_env_orbit(env, agent, 1000, record=True)
#     recording = deepcopy(path['full_state_path'])
#     
#     path2 = _one_agent_env_orbit(env, agent, 500, record=True, playback=recording[500:])
#     recording2 = deepcopy(path2['full_state_path'])
    