# Tract-RLFormer: A Tract-Specific RL policy based Decoder-only Transformer Network

An innovative iterative policy learning framework designed for the tract-specific generation of white matter (WM) tracts using GPT in reinforcement learning (RL) space.

## Getting started

### Installation

It is recommended to use `virtualenv` to run the code

``` bash
virtualenv .env --python
source .env/bin/activate
```

``` bash
# Install common requirements

# edit requirements.txt as needed to change your torch install
pip install -r requirements.txt
# Install some specific requirements directly from git
# scilpy 1.3.0 requires a deprecated version of sklearn on pypi
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install git+https://github.com/scilus/scilpy@1.3.0#egg=scilpy
pip install git+https://github.com/scil-vital/dwi_ml#egg=dwi_ml
pip install git+https://github.com/scilus/ismrm_2015_tractography_challenge_scoring.git
# Load the project into the environment
pip install -e .
```

### Training - T-RLF

For training T-RLF model, tract dataset containing trajectories of RL model on trainset is required.
The datasets for both pretraining and tract-specific finctuning are available here:

#### Pretraining:
https://zenodo.org/records/13383557

https://zenodo.org/records/13383814

https://zenodo.org/records/13383834

#### Finetuning:
https://zenodo.org/records/13382206

Example pretraining and finetuning scripts are available in `scripts` folder.
Otherwise training can be performed by running `T-RLF/experiment.py`

```bash
usage: experiment.py [-h] [--env ENV] [--dataset DATASET] [--mode MODE]
                     [--K K] [--pct_traj PCT_TRAJ] [--batch_size BATCH_SIZE]
                     [--model_type MODEL_TYPE] [--embed_dim EMBED_DIM]
                     [--n_layer N_LAYER] [--n_head N_HEAD]
                     [--activation_function ACTIVATION_FUNCTION]
                     [--dropout DROPOUT] [--learning_rate LEARNING_RATE]
                     [--weight_decay WEIGHT_DECAY]
                     [--warmup_steps WARMUP_STEPS]
                     [--num_eval_episodes NUM_EVAL_EPISODES]
                     [--max_iters MAX_ITERS]
                     [--num_steps_per_iter NUM_STEPS_PER_ITER]
                     [--device DEVICE] [--log_to_wandb LOG_TO_WANDB]
                     [--load_model LOAD_MODEL] [--model_path MODEL_PATH]
                     [--save_path SAVE_PATH]
                     [--save_sc_params_path SAVE_SC_PARAMS_PATH]
                     [--load_usigma LOAD_USIGMA] [--usigma_data USIGMA_DATA]
                     [--from_pretrained FROM_PRETRAINED]
                     [--pretrained_path PRETRAINED_PATH] --env_config
                     ENV_CONFIG
```

### Inference - Tracking

Example tracking script is available in `scripts` folder.

For tracking, you need a trained model. Tract-specific trained models for TD3 and T-RLF are available in `trained_models` folder. Tracking can be then performed by running `T-RLF/inference_tracking.py`
```bash 
usage: inference_tracking.py [-h] [--is_rl IS_RL]
                             [--rl_model_load_path RL_MODEL_LOAD_PATH]
                             [--trlf_model_load_path TRLF_MODEL_LOAD_PATH]
                             [--offline_trajectories OFFLINE_TRAJECTORIES]
                             [--load_sc_params_path LOAD_SC_PARAMS_PATH]
                             [--max_episode_return MAX_EPISODE_RETURN]
                             [--scale_rtg SCALE_RTG]
                             [--timesteps_embed_trlf_model_dim TIMESTEPS_EMBED_TRLF_MODEL_DIM]
                             [--input_fodf_signal INPUT_FODF_SIGNAL]
                             [--target_sh_order TARGET_SH_ORDER]
                             [--seeding_mask SEEDING_MASK]
                             [--tracking_mask TRACKING_MASK]
                             [--bundle_mask BUNDLE_MASK] [--peaks PEAKS]
                             [--step_size STEP_SIZE] [--theta THETA]
                             [--npv NPV] [--min_length MIN_LENGTH]
                             [--max_length MAX_LENGTH]
                             [--reference_file_fa REFERENCE_FILE_FA]
                             [--voxel_size VOXEL_SIZE]
                             [--tracking_batch_size TRACKING_BATCH_SIZE]
                             [--save_trk_path SAVE_TRK_PATH] [--device DEVICE]
```

## Contributing

Contributions are encouraged! If you find any TODOs throughout the code that might spark your interest. There’s ample opportunity to enhance the code's architecture by reorganizing, refactoring, and refining it for better clarity. Additionally, there are several straightforward ways to boost performance.

## Citing Tract-RLFormer

If you use Tract-RLFormer in your research, please cite it as follows:

```
@article{<TBD>,
  title={Tract-RLFormer: A Tract-Specific RL policy based Decoder-only Transformer Network},
  author={Ankita Joshi, Ashutosh Sharma, Anoushkrit Goel, Ranjeet Ranjan Jha, Chirag Ahuja, Arnav Bhavsar, and Aditya Nigam},
  journal={<TBD>},
  year={2024}
}
```