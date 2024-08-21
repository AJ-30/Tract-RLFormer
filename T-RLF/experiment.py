import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from tractRLformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from tractRLformer.models.decision_transformer import DecisionTransformer
from tractRLformer.models.mlp_bc import MLPBCModel
from tractRLformer.training.act_trainer import ActTrainer
from tractRLformer.training.seq_trainer import SequenceTrainer

# Random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    #addition by ashutosh------------------------------------------------------------
    elif env_name == 'ttl2':
        # sys.path.append('/home/turing/TrackToLearn-2/CustomTracking_AS')
        from rl_environments.BaseEnvmod_TRLF import TrackingEnvironment
        import nibabel as nib
        env_dto = {
            'interface_seeding': False,
            'fa_map': None,
            'n_signal': 1,
            'n_dirs': 4,
            'step_size': 0.75,
            'theta': 60,
            'min_length': 0.0,
            'max_length': 200.0,
            'cmc': False,
            'asymmetric': False,
            'prob': 0.0,
            'npv': 7,
            'rng': np.random.RandomState(seed=1111),
            'scoring_data': None,#'/neuro/Tractography_ankita/ankita_ismrm2015/scoring_data',
            'reference': '/neuro/Tractography_ankita/ankita_ismrm2015/raw_data/fa.nii.gz',##'/datasets/TractoInferno-ds003900/derivatives/trainset/sub-1005/dti/sub-1005__fa.nii.gz',#'/scratch/ashutoshs.cse.iitmandi/work/datasets/TractoInferno-ds003900/derivatives/trainset/sub-1005/dti/sub-1005__fa.nii.gz',#'/neuro/Tractography_ankita/ankita_ismrm2015/raw_data/fa.nii.gz',
            'alignment_weighting': 1,
            'straightness_weighting': 0,
            'length_weighting': 0,
            'target_bonus_factor': 0,
            'exclude_penalty_factor': 0,
            'angle_penalty_factor': 0,
            'add_neighborhood': 0.75,
            'compute_reward': True,
            'device': 'cuda'
        }
        """
        inputs
        """
        from dipy.data import get_sphere
        from scilpy.reconst.utils import get_sh_order_and_fullness
        from dipy.reconst.csdeconv import sph_harm_ind_list
        from scilpy.reconst.multi_processes import convert_sh_basis
        #---------
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
        signal = '/neuro/Tractography_ankita/ankita_ismrm2015/raw_data/fodf.nii.gz'##'/datasets/TractoInferno-ds003900/derivatives/trainset/sub-1005/fodf/sub-1005__fodf.nii.gz'#'/scratch/ashutoshs.cse.iitmandi/work/datasets/TractoInferno-ds003900/derivatives/trainset/sub-1005/fodf/sub-1005__fodf.nii.gz'
        signal = nib.load(signal)
        if not np.allclose(np.mean(signal.header.get_zooms()[:3]),signal.header.get_zooms()[0], atol=1e-03):
            print('WARNING: ODF SH file is not isotropic. Tracking cannot be '
                        'ran robustly. You are entering undefined behavior '
                        'territory.')
            
        data = set_sh_order_basis(signal.get_fdata(dtype=np.float32),
                                        'descoteaux07',
                                        target_order=8,#8,
                                        target_basis='descoteaux07')

        seeding_mask = nib.load('/datasets/ismrm_resampled_masks/Cingulum_right_resampled.nii.gz')##'/datasets/TractoInferno-ds003900/pop_avg-aligned-masks/FPT_R/sub-1005_aligned.nii.gz')

        tracking_mask = nib.load('/datasets/ismrm_resampled_masks/Cingulum_right_resampled.nii.gz')##'/datasets/TractoInferno-ds003900/pop_avg-aligned-masks/FPT_R/sub-1005_aligned.nii.gz')

        wm = nib.load('/datasets/ismrm_resampled_masks/Cingulum_right_resampled.nii.gz')##'/datasets/TractoInferno-ds003900/pop_avg-aligned-masks/FPT_R/sub-1005_aligned.nii.gz')
        wm_data = wm.get_fdata()

        if len(wm_data.shape) == 3:
            wm_data = wm_data[..., None]

        signal_data = np.concatenate([data, wm_data], axis=-1)
        signal_data = nib.Nifti1Image(signal_data, affine=signal.affine)

        peaks = nib.load('/neuro/Tractography_ankita/ankita_ismrm2015/raw_data/peaks.nii.gz')##"/datasets/TractoInferno-ds003900/derivatives/trainset/sub-1005/fodf/sub-1005__peaks.nii.gz")
        #----------
        env = TrackingEnvironment(
                        input_volume=signal_data,
                        tracking_mask=tracking_mask,
                        target_mask=None,
                        seeding_mask=seeding_mask,
                        peaks=peaks,
                        env_dto=env_dto
                    )
    
        max_ep_len = 530
        env_targets = [100, 200, 300, 400, 500]  
        scale = 100.
        #-------------------------------------------------------------------------------
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations
    #CHANGED BY ASHUTOSH ----------------------------env.state_shape[0]-----------------------
    if env_name == 'ttl2':
        state_dim = env.get_state_size()
        # print(state_dim)
        act_dim = env.get_action_size()

    else:
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    #--------------------------------------------------------------------------------------------

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    #ADDED BY ASHUTOSH------------------------------------------------------------------------
    if(env_name == 'ttl2'):
        dataset_path = dataset
    #--------------------------------------------------------------------------------------------
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    #For input normalization, use different file if specified
    if variant['load_usigma'] == 1:
        with open(variant['usigma_data'], 'rb') as f:
            usigma_traj = pickle.load(f)
        states_usigma = []
        for path in usigma_traj:
            states_usigma.append(path['observations'])
        states_usigma = np.concatenate(states_usigma, axis=0)
        state_mean, state_std = np.mean(states_usigma, axis=0), np.std(states_usigma, axis=0) + 1e-6
        print('usigma normalization. .. ...')
        del states_usigma
        del usigma_traj
    else:
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    #----------------------------------------scale-mean-std
    # Define scale parameters for mean and std 
    scale_mean = torch.nn.Parameter(torch.ones(state_dim, device=device))
    scale_std = torch.nn.Parameter(torch.ones(state_dim, device=device))
    # if variant['load_sc_param']:
    #     checkpoint = torch.load(variant.get('load_sc_param_path', 'scale_parameters.pth'), map_location='cpu')
    #     scale_mean = checkpoint['scale_mean'].to(device)
    #     scale_std = checkpoint['scale_std'].to(device)
    #----------------------------------------scale-mean-std

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        # print(f'\n\np_sample: {p_sample}\n Batch_indices: {batch_inds} \n\n')


        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(torch.from_numpy(traj['observations'][si:si + max_len]).to(device=device, dtype=torch.float32).reshape(1, -1, state_dim))#traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # print(f'{s[-1].shape}\n\n\n')
            s[-1] = torch.cat([torch.zeros((1, max_len - tlen, state_dim), device=device), s[-1]], dim=1)#np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            #----------------------------------------scale-mean-std
            # s[-1] = (s[-1] - state_mean) / state_std #ORIGINAL WAS THIS
            state_mean_tensor = torch.tensor(state_mean, dtype=torch.float32, device=device)
            state_std_tensor = torch.tensor(state_std, dtype=torch.float32, device=device)

            # Convert s[-1] to a tensor on the same device
            # s_tensor = torch.from_numpy(s[-1]).to(device)

            # Perform scaling and normalization:
            s[-1] = (s[-1] - state_mean_tensor * scale_mean) / (state_std_tensor * scale_std)
            # s_tensor = (s_tensor - state_mean_tensor * scale_mean) / (state_std_tensor * scale_std)

            # s[-1] = s_tensor#s_tensor.detach().cpu().numpy()

            # print(f'ye raheeee is batch ke... {scale_mean, scale_std}\n\n')
            #----------------------------------------scale-mean-std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.cat(s, dim=0)#torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        # print(s[...,0], 'STATE haiii, \n\n', s.shape, '\n\n')
        # print(f'timesteps hai.. {timesteps}\n\n')
        # print(f'rtg hai.. {rtg.shape}\n\n')

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError
    #ASHUTOSH------------------------------------------------------------------------------
    load_model = variant.get('load_model', False)
    model_path = variant.get('model_path', 'dt_Fornix_512.pt')
    if load_model:
        model = torch.load(model_path)
        print(f'Model Loaded.... {model_path}')
    #--------------------------------------------------------------------------------------------------------
    #for finetuning----------------------------
    is_from_pretrained = variant.get('from_pretrained', False)
    if is_from_pretrained:
        #making new decoder block
        new_decoder_block = model
        new_decoder_block = new_decoder_block.transformer.h[-1]

        pretrained_path = variant['pretrained_path']
        model = torch.load(pretrained_path, map_location='cpu')
        model = model.to(device)

        # for block in model.transformer.h:
        #     for param in block.parameters():
        #         param.require_grad = False
        for param in model.parameters():
            param.requires_grad = False

        print(model.transformer.config.n_layer)
        
        if model.transformer.config.n_layer != 4:
            model.transformer.h.append(new_decoder_block)
            print("\nModel is now 4-decoder block model\n")
        else:
            print("\nModel already has 4 layers, so making only last layer as learnable.\n")
            for param in model.transformer.h.parameters():
                param.require_grad = True
        # model.transformer.set_layers(4)
        # model.transformer.use_layers = 4
        model.transformer.config.n_layer = 4
        # print('\n\n',model.transformer.use_layers,'\n\n')
        # print("\nModel is now 4-decoder block model\n")
    #for finetuning----------------------------

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    #----------------------------------------scale-mean-std
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=variant['learning_rate'],
    #     weight_decay=variant['weight_decay'],
    # ) #ORIGINAL DT OPTIMIZER
    # Add scale parameters to the list of parameters to optimize
    if is_from_pretrained:
        parameters_to_optimize = list(model.transformer.h[-1].parameters())
        print("\nOnly last decoder block is set as trainable...\n")
    else:
        parameters_to_optimize = list(model.parameters()) + [scale_mean, scale_std]
    optimizer = torch.optim.AdamW(
        parameters_to_optimize,
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    #----------------------------------------scale-mean-std
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),#------------------mod by AJ
            loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: 180/np.pi*(torch.acos(torch.clamp(torch.dot(a.view(-1)/torch.norm(a), a_hat.view(-1)/torch.norm(a_hat)), -1.0, 1.0))),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),#------------------- mod by AJ
            loss_fn = lambda s_hat, a_hat, r_hat, s, a, r: 180/np.pi*(torch.acos(torch.clamp(torch.dot(a/torch.norm(a), a_hat/torch.norm(a_hat)), -1.0, 1.0))),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    # with torch.autograd.set_detect_anomaly(True):
        
    # if np.max(returns) > 1000:
    #     variant['max_iters'] = 20
    mean_losses_train = []
    consecutive_non_decreasing = 0
    patience = 3
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        curr_loss = outputs['training/train_loss_mean']
        if log_to_wandb:
            wandb.log(outputs)
        mean_losses_train.append(curr_loss)

        # Check for loss convergence
        if len(mean_losses_train) > 1 and mean_losses_train[-2] - mean_losses_train[-1] <= 1e-2:
            consecutive_non_decreasing += 1
        else:
            consecutive_non_decreasing = 0

        if consecutive_non_decreasing >= patience:
            break

        if iter >=10 and iter%10==0:#save model after iteration, to compare results, overfitting, optimal loss/iterations
            save_path = variant.get('save_path', 'base_model_fpt-r_1102_full.pt')
            torch.save(model, f'{save_path[:-3]}_{iter}.pt')
            save_sc_param = variant.get('save_sc_params_path','/home/turing/TrackToLearn-2/decision-transformer/gym/dt_hyperparams_search/results/sc_params/scale_parameters.pth')
            torch.save({'scale_mean': scale_mean, 'scale_std': scale_std}, f'{save_sc_param[:-4]}_{iter}.pth')

    # Save trained model

    save_path = variant.get('save_path', 'base_model_fpt-r_1102_full.pt')
    torch.save(model, save_path) #2_17 tak kar lia tha
    # Saving scale parameters
    torch.save({
        'scale_mean': scale_mean,
        'scale_std': scale_std
    }, variant.get('save_sc_params_path','sc_params/scale_parameters.pth'))
    print(f'{save_path} Model saved...\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='dt_Fornix_512.pt')
    parser.add_argument('--save_path', type=str, default='base_model.pt')
    parser.add_argument('--save_sc_params_path', type=str, default='sc_params/scale_parameters.pth')
    parser.add_argument('--load_usigma', type=int, default=0)
    parser.add_argument('--usigma_data', type=str, default='sc_params/scale_parameters.pth')
    #for finetuning--------------
    parser.add_argument('--from_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='trlf_Fornix_512.pt')

    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))