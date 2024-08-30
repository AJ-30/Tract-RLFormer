import argparse
from rl_environments.BaseEnvmod_TRLF import *

    
def track_main(
    variant
):
    my_device = variant.get('device', 'cuda')

    
    import torch.nn as nn
    import nibabel as nib
    import numpy as np
    import torch 

    from torch.nn import DataParallel #for multiGPU tracking

    torch.cuda.empty_cache()

    # # Load model
    
    import sys
    from rl_environments.td3 import TD3

    from decision_transformer.models.decision_transformer import DecisionTransformer

    rl_flag = 0
    trlf_flag = 0


    torch.cuda.is_available()

    is_rl = variant['is_rl']
    if is_rl == True:
        rl_flag = 1
        trlf_flag = 0
        
        td3_model = TD3(334, 3, 1024, rng=np.random.RandomState(seed=1111), device = my_device) #TODO: change this later
        rl_path = variant.get('rl_model_load_path', 'td3')
        td3_model.policy.load(rl_path, 'last_model_state')
        print("tracking using RL...")

    else:
        rl_flag = 0
        trlf_flag = 1
        trlf_model_path = variant.get('trlf_model_load_path', 'T_models/M3_D3_3k-3k_sub-1005-1009_sub-1011.pt') #variant.get('device', 'cuda')
        trlf_model = torch.load(trlf_model_path, map_location=torch.device(my_device))
        
        trlf_model.eval()
        trlf_model.to(device=my_device)


        print(f'model is {trlf_model_path}')

    import pickle
    import tqdm
    from dipy.tracking.streamlinespeed import compress_streamlines
    from nibabel.streamlines.tractogram import LazyTractogram
    from nibabel.streamlines.tractogram import TractogramItem

    def mean_var_offline(pkl_file):
        with open(pkl_file, 'rb') as file:
            trajectories = pickle.load(file)
        
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        return state_mean, state_std

    if trlf_flag == 1:
        offline_traj = variant.get('offline_trajectories', 'base_model_FPT_R_1102_sample_random.pkl')
        st_mean, st_std = mean_var_offline(offline_traj)

        # Loading scale parameters for mean and standard deviation
        checkpoint = torch.load(variant.get('load_sc_params_path', 'scale_parameters.pth'), map_location='cpu')
        # checkpoint.to(my_device)
        scale_mean = checkpoint['scale_mean']
        scale_std = checkpoint['scale_std']
        st_mean *= scale_mean.detach().cpu().numpy()
        st_std *= scale_std.detach().cpu().numpy()

        state_mean = torch.from_numpy(st_mean).to(device=my_device, dtype=torch.float32)
        state_std = torch.from_numpy(st_std).to(device=my_device, dtype=torch.float32)

    def model_validation_episode(
        initial_state,
        env: BaseEnv,
        compress=False,
        buffer_list=[],
        end=-1
    ):
        """
        Main loop for the algorithm
        From a starting state, run the model until the env. says its done

        Parameters
        ----------
        initial_state: np.ndarray
            Initial state of the environment
        env: BaseEnv
            The environment actions are applied on. Provides the state fed to
            the RL algorithm

        Returns
        -------
        tractogram: Tractogram
            Tractogram containing the tracked streamline
        running_reward: float
            Cummulative training steps reward
        """

        running_reward = 0
        state = initial_state
        done = False

        if trlf_flag:
            states = state.reshape(state.shape[0], 1, state.shape[1]).to(device=my_device, dtype=torch.float32)
            actions = torch.zeros((0, 1, env.get_action_size()), device=my_device, dtype=torch.float32)
            rewards = torch.zeros(0, device=my_device, dtype=torch.float32)
            ep_return = [variant.get('max_episode_return', 300)/variant.get('scale_rtg', 100) for _ in range(states.shape[0])]#target_return #CHANGED TO 300 BECAUSE DONT WANT NEGATIVE AS SOME TRAJ COME OVER 200
            target_return = torch.tensor(ep_return, device=my_device, dtype=torch.float32).reshape(states.shape[0], 1)
            t = 0
            l = [t for _ in range(states.shape[0])]
            timesteps = torch.tensor(l, device=my_device, dtype=torch.long).reshape(states.shape[0], 1)
            
        #----------------------------------------------------

        while not np.all(done):
            # Select action according to policy + noise to make tracking
            # probabilistic
            
            if rl_flag:
                action = td3_model.policy.select_action(state)
            

            
            if trlf_flag:
                if actions.shape[0] == 0:
                    actions = torch.cat([actions, torch.zeros((states.shape[0], 1,  env.get_action_size()), device=my_device)], dim=0)
                else:
                    actions = torch.cat([actions, torch.zeros((states.shape[0], 1, env.get_action_size()), device=my_device)], dim=1)
                rewards = torch.cat([rewards, torch.zeros((states.shape[0], 1), device=my_device)], dim=1)

                with torch.no_grad():
                    action = trlf_model.get_action(
                        (states.to(dtype=torch.float32) - state_mean) / state_std,
                        actions.to(dtype=torch.float32),
                        rewards.to(dtype=torch.float32),
                        target_return.to(dtype=torch.float32),
                        timesteps,
                    )
            
                actions[:,-1] = action
                
                action = action.reshape(states.shape[0],3).detach().cpu().numpy() #CHECK once

            

            # Perform action
            next_state, reward, done, *_ = env.step(action)

            
            if trlf_flag:
                rewards[:,-1] = torch.tensor(reward, device=my_device, dtype=torch.float32)
                
                cur_state = next_state.to(device=my_device).reshape(states.shape[0],1, state.shape[1])
                states = torch.cat([states, cur_state], dim=1)
                
                #now for rtg
                pred_return = target_return[:, -1] - (torch.tensor(reward, device=my_device, dtype=torch.float32)/variant.get('scale_rtg', 100)) #RTG is scaled by 100 during training
                
                target_return = torch.cat(
                    [target_return, pred_return.reshape(states.shape[0], 1)], dim=1)
                
                if t+1 < variant.get('timesteps_embed_trlf_model_dim', 100):
                    timesteps = torch.cat(
                        [timesteps,
                    torch.ones((states.shape[0], 1), device=my_device, dtype=torch.long) * (t+1)], dim=1)
                else:
                    timesteps = torch.cat(
                        [timesteps,
                    torch.ones((states.shape[0], 1), device=my_device, dtype=torch.long) * (variant.get('timesteps_embed_trlf_model_dim', 100) -1) ], dim=1)
                t+=1 #timestep increase everytime
                

            to_prune = []

            for i in range(len(done)):
                    
                if done[i] == True:
                    to_prune.append(i)

            to_prune = torch.tensor(to_prune)  # Index list


            if trlf_flag == 1:
                unique_indices = np.setdiff1d(np.arange(actions.shape[0]), to_prune)
                unique_indices = torch.tensor(unique_indices, device=my_device).long()
                selected_actions = torch.index_select(actions, dim=0, index=unique_indices)
                del actions
                actions = selected_actions
                del selected_actions
                
                selected_rewards = torch.index_select(rewards, dim=0, index=unique_indices)
                del rewards
                rewards = selected_rewards
                del selected_rewards
                
                selected_target_return = torch.index_select(target_return, dim=0, index=unique_indices)
                del target_return
                target_return = selected_target_return
                del selected_target_return
                
                selected_timesteps = torch.index_select(timesteps, dim=0, index=unique_indices)
                del timesteps
                timesteps = selected_timesteps
                del selected_timesteps
                
                selected_states = torch.index_select(states, dim=0, index=unique_indices)
                del states
                states = selected_states
                del selected_states


            # Keep track of reward
            running_reward += sum(reward)

            # "Harvesting" here means removing "done" trajectories
            # from state. This line also set the next_state as the
            # state
            state, _ = env.harvest(next_state)

        return running_reward


    # ### tracker class

    
    #Updated Tracker class:

    class Tracker(object):
        """ Tracking class similar to scilpy's or dwi_ml's. This class is
        responsible for generating streamlines, as well as giving back training
        or RL-associated metrics if applicable.
        """

        def __init__(
            self,
            alg, ##
            env: BaseEnv,
            back_env: BaseEnv,
            n_actor: int,
            interface_seeding: bool,
            no_retrack: bool,
            compress: float = 0.0,
            save_seeds: bool = False
        ):
            """

            Parameters
            ----------
            alg: Either RLAlgorithm or DT
                Tracking agent.
            env: BaseEnv
                Forward environment to track.
            back_env: BaseEnv
                Backward environment to track.
            compress: float
                Compression factor when saving streamlines.

            """

            self.alg = alg
            self.env = env
            self.back_env = back_env
            self.n_actor = n_actor
            self.interface_seeding = interface_seeding
            self.no_retrack = no_retrack
            self.compress = compress
            self.save_seeds = save_seeds

        def track(
            self,
        ):
            """ Actual tracking function. Use this if you just want streamlines.

            Track with a generator to save streamlines to file
            as they are tracked. Used at tracking (test) time. No
            reward should be computed.

            Returns:
            --------
            tractogram: Tractogram
                Tractogram in a generator format.

            """

            # Presume iso vox
            vox_size = abs(self.env.affine[0][0])

            compress_th_vox = self.compress / vox_size

            batch_size = self.n_actor

            # Shuffle seeds so that massive tractograms wont load "sequentially"
            # when partially displayed
            np.random.shuffle(self.env.seeds)

            def tracking_generator():
                # Switch policy to eval mode so no gradients are computed
                if rl_flag:
                    self.alg.policy.eval()
                if trlf_flag:
                    self.alg.eval()
                # Track for every seed in the environment
                for i, start in enumerate(tqdm.tqdm(range(0, len(self.env.seeds), batch_size))):

                    # Last batch might not be "full"
                    end = min(start + batch_size, len(self.env.seeds))

                    state = self.env.reset(start, end)

                    # Track forward
                    model_validation_episode(state, self.env, end = 412002)

                    batch_tractogram = self.env.get_streamlines()

                    if not self.interface_seeding:
                        state = self.back_env.reset(batch_tractogram.streamlines)

                        # Track backwards
                        model_validation_episode(state, self.back_env, end = 832002)

                        batch_tractogram = self.back_env.get_streamlines()

                    for item in batch_tractogram:

                        streamline_length = len(item)

                        streamline = item.streamline
                        streamline += 0.5 #
                        streamline *= vox_size

                        seed_dict = {}
                        if self.save_seeds:
                            seed = item.data_for_streamline['seeds']
                            seed_dict = {'seeds': seed-0.5} #

                        if self.compress:
                            streamline = compress_streamlines(streamline, compress_th_vox) #THIS joins the co-linear points

                        if (self.env.min_nb_steps < streamline_length < self.env.max_nb_steps):
                            yield TractogramItem(streamline, seed_dict, {})

            tractogram = LazyTractogram.from_data_func(tracking_generator)

            return tractogram

    # # Tracking now...

    from dipy.data import get_sphere
    from scilpy.reconst.utils import get_sh_order_and_fullness
    from dipy.reconst.csdeconv import sph_harm_ind_list
    from scilpy.reconst.multi_processes import convert_sh_basis

    
    def set_sh_order_basis(
        sh,
        sh_basis,
        target_basis='descoteaux07',
        target_order=6,
        sphere_name='repulsion724',
    ):
        """ Convert SH to the target basis and order. In practice, it is always
        order 6 and descoteaux07 basis.

        This uses a lot of "hacks" to convert the ODFs. To go from full to
        symmetric basis, only even coefficents are selected.

        To go from order N to order 6, SH coefficients are either truncated
        or padded.

        """

        sphere = get_sphere(sphere_name)

        n_coefs = sh.shape[-1]
        sh_order, full_basis = get_sh_order_and_fullness(n_coefs)
        sh_order = int(sh_order)

        # If SH in full basis, convert them
        if full_basis is True:
            print('SH coefficients are in "full" basis, only even coefficients '
                'will be used.')
            _, orders = sph_harm_ind_list(sh_order, full_basis)
            sh = sh[..., orders % 2 == 0]

        # If SH are not of order 6, convert them
        if sh_order != target_order:
            print('SH coefficients are of order {}, '
                'converting them to order {}.'.format(sh_order, target_order))
            target_n_coefs = len(sph_harm_ind_list(target_order)[0])

            if n_coefs > target_n_coefs:
                sh = sh[..., :target_n_coefs]
            else:
                X, Y, Z = sh.shape[:3]
                n_missing_coefs = target_n_coefs - n_coefs
                sh = np.concatenate(
                    (sh, np.zeros((X, Y, Z, n_missing_coefs))), axis=-1)

        # If SH are not in the descoteaux07 basis, convert them
        if sh_basis != target_basis:
            print('SH coefficients are in the {} basis, '
                'converting them to {}.'.format(sh_basis, target_basis))
            sh = convert_sh_basis(
                sh, sphere, input_basis=sh_basis, nbr_processes=1)

        return sh

    
    #files:
    #input volume:
    signal = variant.get('input_fodf_signal', 'fodf.nii.gz')
    signal = nib.load(signal)
    if not np.allclose(np.mean(signal.header.get_zooms()[:3]),signal.header.get_zooms()[0], atol=1e-03):
        print('WARNING: ODF SH file is not isotropic. Tracking cannot be '
                    'ran robustly. You are entering undefined behavior '
                    'territory.')
        
    data = set_sh_order_basis(signal.get_fdata(dtype=np.float32),
                                    'descoteaux07',
                                    target_order=variant.get('target_sh_order', 8),
                                    target_basis='descoteaux07')

    seeding_mask = nib.load(variant.get('seeding_mask', 'FPT_R.nii.gz'))

    tracking_mask = nib.load(variant.get('tracking_mask', 'FPT_R.nii.gz'))

    wm = nib.load(variant.get('bundle_mask', 'FPT_R.nii.gz'))
    wm_data = wm.get_fdata()

    if len(wm_data.shape) == 3:
        wm_data = wm_data[..., None]

    signal_data = np.concatenate([data, wm_data], axis=-1)
    signal_data = nib.Nifti1Image(signal_data, affine=signal.affine)

    peaks = nib.load(variant.get('peaks', "fodf/sub-1005__peaks.nii.gz"))

    
    tracking_env_dto = {
                # 'dataset_file': self.dataset_file,
                # 'subject_id': self.subject_id,
                'interface_seeding': False,
                'fa_map': None,
                'n_signal': 1,
                'n_dirs': 4,
                'step_size': variant.get('step_size', 0.75),
                'theta': variant.get('theta', 60),
                'min_length': variant.get('min_length', 20.0),
                'max_length': variant.get('max_length', 200.0),
                'cmc': False,
                'asymmetric': False,
                'prob': 0.0,
                'npv': variant.get('npv', 7),
                'rng': np.random.RandomState(seed=1111),
                'scoring_data': None,
                'reference': variant.get('reference_file_fa', 'dti/sub-1005__fa.nii.gz'),
                'alignment_weighting': 1,
                'straightness_weighting': 0,
                'length_weighting': 0,
                'target_bonus_factor': 0,
                'exclude_penalty_factor': 0,
                'angle_penalty_factor': 0,
                'add_neighborhood': 0.75,
                'compute_reward': True,
                'device': my_device
            }

    
    print('Loading environment.')
    env = TrackingEnvironment(
        input_volume=signal_data,
        tracking_mask=tracking_mask,
        target_mask=None,
        seeding_mask=seeding_mask,
        peaks=peaks,
        env_dto=tracking_env_dto
    )


    
    print('Loading back environment.')
    back_env = RetrackingEnvironment(
        env, tracking_env_dto
    )

    action_size = env.get_action_size()

    
    #in model hyperparameters:
    step_size = variant.get('step_size', 0.75)
    voxel_size = variant.get('voxel_size', 2)

    
    tracking_voxel_size = env.get_voxel_size()
    print(tracking_voxel_size)
    step_size_mm = (tracking_voxel_size / voxel_size) * step_size

    print('Step size is this: ', step_size_mm)

    
    if back_env:
        back_env.set_step_size(step_size_mm)
    env.set_step_size(step_size_mm)

    
    # print("Total gpu devices hainnn: ", torch.cuda.device_count())


    if trlf_flag == 1:
        tracker = Tracker(
            trlf_model, env, back_env,
            variant.get('tracking_batch_size', 5000), False, False
        )
    else:
        tracker = Tracker(
            td3_model, env, back_env,
            variant.get('tracking_batch_size', 5000), False, False
        )

    
    tractogram = tracker.track()

    
    tractogram.affine_to_rasmm = env.affine

    
    from dipy.io.utils import get_reference_info, create_tractogram_header

    
    file = variant.get('save_trk_path', 'noGrad/M_10000_BS_1005.trk')
    filetype = nib.streamlines.detect_format(file)
    reference = get_reference_info(variant.get('bundle_mask', 'FPT_R.nii.gz'))
    header = create_tractogram_header(filetype, *reference)

    
    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, file, header=header)

    print(f'Saved to .. {file}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_rl', type=bool, default=False)
    parser.add_argument('--rl_model_load_path', type=str)
    parser.add_argument('--trlf_model_load_path', type=str, default='T_models/M3_D3_3k-3k_sub-1005-1009_sub-1011.pt')
    parser.add_argument('--offline_trajectories', type=str, default='base_model_FPT_R_1102_sample_random.pkl')
    parser.add_argument('--load_sc_params_path', type=str, default='model_checkpoint_mean_std_params.pkl')
    
    parser.add_argument('--max_episode_return', type=int, default=300)
    parser.add_argument('--scale_rtg', type=int, default=100)
    parser.add_argument('--timesteps_embed_trlf_model_dim', type=int, default=100)

    parser.add_argument('--input_fodf_signal', type=str, default='fodf/sub-1005__fodf.nii.gz') 
    parser.add_argument('--target_sh_order', type=int, default=8)
    parser.add_argument('--seeding_mask', type=str, default='FPT_R.nii.gz') 
    parser.add_argument('--tracking_mask', type=str, default='FPT_R.nii.gz') 
    parser.add_argument('--bundle_mask', type=str, default='FPT_R.nii.gz') 
    parser.add_argument('--peaks', type=str, default="fodf/sub-1005__peaks.nii.gz")

    parser.add_argument('--step_size', type=float, default=0.75)
    parser.add_argument('--theta', type=int, default=60)
    parser.add_argument('--npv', type=int, default=7)
    parser.add_argument('--min_length', type=float, default=20.0)
    parser.add_argument('--max_length', type=float, default=200.0)
    parser.add_argument('--reference_file_fa', type=str, default='dti/sub-1005__fa.nii.gz')
    parser.add_argument('--voxel_size', type=float, default=2)

    parser.add_argument('--tracking_batch_size', type=int, default=5000)
    parser.add_argument('--save_trk_path', type=str, default='noGrad/M_10000_BS_1005.trk')

    parser.add_argument('--device', type=str, default='cuda')

    
    args = parser.parse_args()

    track_main(variant=vars(args))

if __name__ == '__main__':
    main()