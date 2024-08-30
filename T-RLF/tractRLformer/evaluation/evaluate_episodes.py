import numpy as np
import torch

#ASHUTOSH -------------TO LOAD STATE FROM TRAJECTORY
import pickle
# pickleFile = open('ttl_traj_list_1_4921.pkl','rb')
# fl = pickle.load(pickleFile)
# init_state = fl[7]['observations'][0]
# #IF 5 STATES FOR START------------------AS
# init_states = fl[7]['observations'][:5]
# init_next_states = fl[7]['next_observations'][:5]
# init_actions = fl[7]['actions'][:5]
# init_rewards = fl[7]['rewards'][:5]
# print(type(init_states))
# print(init_states.shape)
#IF 5 STATES FOR START------------------
#-------------TO LOAD STATE FROM TRAJECTORY

try:
    import sys
    sys.path.append('/home/turing/TrackToLearn-2/CustomTracking_AS')
    from BaseEnvmod_DT import TrackingEnvironment
except ImportError:
    # Handle the error or simply pass
    pass


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


#--------------------------------
def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    #-------------------------------------------------------------------------------------
    if isinstance(env, TrackingEnvironment):
        # seeds = env.seeds
        # print('seeddddds',seeds.shape) # (110106, 3)
        rng = 1998#np.random.randint(1,1998)
        state = env.reset(rng,rng+4)
        state = state.detach().cpu().numpy()
        #IF FROM TRAJECTORY---------
        # state = init_state #CAN'T DO LIKE THIS
        #IF FROM TRAJECTORY---------
        # print(state)
        # print('SHAPE OF STATE: ',state.shape)
    else:#-----------------------------------------------------------------------------------------------
        state = env.reset()
    # state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(state.shape[0],1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0,1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    

    ep_return = [target_return for _ in range(states.shape[0])]#target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(states.shape[0], 1)#torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    l = [0 for _ in range(states.shape[0])]
    timesteps = torch.tensor(l, device=device, dtype=torch.long).reshape(states.shape[0], 1)
    

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        # print('EPISODE LENGTH: ',t, 'max_ep_len:', max_ep_len,'----------------')
        if actions.shape[0] == 0:
            actions = torch.cat([actions, torch.zeros((states.shape[0], 1,  act_dim), device=device)], dim=0)
        else:
            actions = torch.cat([actions, torch.zeros((states.shape[0], 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((states.shape[0], 1), device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        
        actions[:,-1,:] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)
        if isinstance(env, TrackingEnvironment):#
            state = state.detach().cpu().numpy()#
            reward = reward[0]#
        

        # print('evalllll-----',state.shape)
        cur_state = torch.from_numpy(state).to(device=device).reshape(states.shape[0], 1, state_dim)
        states = torch.cat([states, cur_state], dim=1)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[:,-1] - (reward/scale)
        else:
            pred_return = target_return[:,-1]

        target_return = torch.cat(
            [target_return, pred_return.reshape(states.shape[0], 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((states.shape[0], 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if np.all(done):
            break

    return episode_return, episode_length
