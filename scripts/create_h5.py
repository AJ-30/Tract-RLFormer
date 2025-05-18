#%%
import sys
import numpy as np
import nibabel as nib
import os
import subprocess
#%%

ttoi_path = 'TractoInferno-ds003900/derivatives/'
sys.path.append('TrackToLearn')

#%%

def get_subject_ids(folder, split):
    subjects_path = os.path.join(folder, split)
    if not os.path.exists(subjects_path):
        raise FileNotFoundError(f"The path '{subjects_path}' does not exist. Only split 'trainset' or 'testset' or 'validset' are allowed.")
    subjects_list = os.listdir(subjects_path)
    return subjects_list


splits = ['trainset', 'validset', 'testset']
all_subs_dict = {split: get_subject_ids(ttoi_path, split) for split in splits}



#%%
sys.path.append('TrackToLearn')

def find_key_by_element(dictionary, target_element):
    for key, value_list in dictionary.items():
        if target_element in value_list:
            return key
    return None

# %%
sys.path.append('/home/turing/TrackToLearn')

def make_h5_from_ids(all_subs_dict, ttoi_path, subs_list):
    params_list = ['fodf', 'wm', 'peaks', 'sub_name', 'h5_loc', 'gm', 'csf']
    params_dict = dict.fromkeys(params_list)
    script = 'TrackToLearn/datasets/create_dataset_wo_wm.py'
    for sub_id in subs_list:
        split = find_key_by_element(all_subs_dict, sub_id)
        base_path = ttoi_path+split+'/'+sub_id+'/'
        params_dict['fodf'] = os.path.join(base_path+'/fodf/', sub_id+'__fodf.nii.gz')
        params_dict['peaks'] = os.path.join(base_path+'/fodf/', sub_id+'__peaks.nii.gz')
        params_dict['gm'] = os.path.join(base_path+'/mask/', sub_id+'__mask_gm.nii.gz')
        params_dict['csf'] = os.path.join(base_path+'/mask/', sub_id+'__mask_csf.nii.gz')
        params_dict['h5_loc'] = base_path
        params_dict['sub_name'] = sub_id
        if not os.path.exists(params_dict['h5_loc']+'/'+params_dict['sub_name']+'.hdf5'):
            subprocess.run(["python", script, f'{params_dict["fodf"]}', f'{params_dict["peaks"]}', f'{params_dict["sub_name"]}', f'{params_dict["sub_name"]}', f'{params_dict["h5_loc"]}', f'--gm', f'{params_dict["gm"]}', f'--csf', f'{params_dict["csf"]}'])
            print('saved hdf5 for '+ sub_id+ ' at loc '+params_dict['h5_loc']+params_dict['sub_name']+'.hdf5')
        else:
            print('skipping... hdf5 already exists for '+sub_id)


#%%
subs_list = ['sub-1030', 'sub-1079', 'sub-1180', 'sub-1198']
make_h5_from_ids(all_subs_dict, ttoi_path, subs_list)
