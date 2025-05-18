import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import os

from dipy.data import get_sphere
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.reconst.csdeconv import sph_harm_ind_list
from scilpy.reconst.multi_processes import convert_sh_basis

import argparse

        
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(322, 512)  # Increase size of the first hidden layer
        self.norm1 = nn.BatchNorm1d(512)  # Batch normalization
        self.dropout1 = nn.Dropout(0.5)   # Dropout layer
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 128)
        self.norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.norm1(x)  # Apply batch normalization
        x = self.dropout1(x)  # Apply dropout
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

#methods related to getting subjects, loading ttoi data
def get_subject_ids(folder, split):
    subjects_path = os.path.join(folder, split)
    if not os.path.exists(subjects_path):
        raise FileNotFoundError(f"The path '{subjects_path}' does not exist. Only split 'trainset' or 'testset' or 'validset' are allowed.")
    subjects_list = os.listdir(subjects_path)
    return subjects_list

def find_key_by_element(dictionary, target_element):
    for key, value_list in dictionary.items():
        if target_element in value_list:
            return key
    return None


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

def validation(variant):

    bundle = variant.get('bundle')
    bundle_new = variant.get('bundle_new')
    sub = variant.get('subject')
    is_hcp = variant.get('hcp')
    is_hcp_new = variant.get('hcp_new')
    is_ttoi_new = variant.get('ttoi_new')
    is_ismrm = variant.get('ismrm')

    ttoi_path = 'TractoInferno-ds003900/derivatives/'
    splits = ['trainset', 'validset', 'testset']
    all_subs_dict = {split: get_subject_ids(ttoi_path, split) for split in splits}

    if not is_hcp and not is_ismrm:
        split = find_key_by_element(all_subs_dict, sub)

    if is_hcp_new:
        hcp_path = '105HCP/'
        splits = ['train', 'val', 'test']
        all_subs_dict = {split: get_subject_ids(hcp_path, split) for split in splits}
        split = find_key_by_element(all_subs_dict, sub)


    def compute_voxel_pairwise_measures_4masks(bundle_binary_map, gs_binary_map):
        """Compute comparison measures between two masks."""

        bundle_indices = np.where(bundle_binary_map.flatten() > 0)[0]
        gs_indices = np.where(gs_binary_map.flatten() > 0)[0]

        tp = len(np.intersect1d(bundle_indices, gs_indices))
        fp = len(np.setdiff1d(bundle_indices, gs_indices))
        fn = len(np.setdiff1d(gs_indices, bundle_indices))

        if tp == 0:
            overlap = 0.
            overreach = None
            dice = 0.
        else:
            overlap = tp / float(tp + fn)
            overreach = fp / float(tp + fn)
            dice = 2 * tp / float(2 * tp + fp + fn)

        return {"dice": dice, "overlap": overlap, "overreach": overreach}


    if is_hcp_new or is_ttoi_new:
        save_path = 'mask_refinement_module/output_masks/'+bundle_new+'/'
    else:
        save_path = 'mask_refinement_module/output_masks/'+bundle+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def generate_new_mask(model, signal_data, tracking_approximated_mask):
        # Ensure the model is in evaluation mode
        model.eval()

        # Offsets for neighbor info
        offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

        # Create an empty array to store the generated mask
        tracking_approximated_mask_data = tracking_approximated_mask.get_fdata()
        generated_approximated_mask = np.zeros_like(tracking_approximated_mask_data)

        # Iterate over each voxel
        for i in range(tracking_approximated_mask_data.shape[0]):
            for j in range(tracking_approximated_mask_data.shape[1]):
                for k in range(tracking_approximated_mask_data.shape[2]):
                    # Check if the voxel is within the tracking_approximated_mask
                    if tracking_approximated_mask_data[i, j, k] == 1:
                        # Extract features for the current voxel and its neighbors
                        features = []
                        features.extend(signal_data.get_fdata()[i, j, k, :])
                        for offset in offsets:
                            ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]
                            if 0 <= ni < tracking_approximated_mask_data.shape[0] and \
                            0 <= nj < tracking_approximated_mask_data.shape[1] and \
                            0 <= nk < tracking_approximated_mask_data.shape[2]:
                                features.extend(signal_data.get_fdata()[ni, nj, nk, :])
                            else:
                                features.extend([0] * signal_data.shape[-1])

                        # Convert features to tensor and move to device
                        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)#.to('cuda')

                        # Forward pass through the model
                        with torch.no_grad():
                            output = model(features_tensor)

                        # Predicted value (probability)
                        prediction = output.item()

                        # Replace the value in the generated mask based on prediction
                        generated_approximated_mask[i, j, k] = 1 if prediction >= 0.5 else 0

        # Convert the generated mask to Nifti format
        generated_approximated_mask_nifti = nib.Nifti1Image(generated_approximated_mask, affine=tracking_approximated_mask.affine)

        # Save the generated mask to a file
        nib.save(generated_approximated_mask_nifti, f'{save_path}{sub}-generated_approximated_mask.nii.gz')

        return generated_approximated_mask_nifti

    model = SimpleNN()#.to('cuda')

    # Load the saved model state dictionary
    saved_model_path = variant.get('saved_model_path')
    model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))


    if is_ismrm:
        signal = 'ismrm2015/raw_data/fodf_45.nii.gz'
    elif is_hcp:
        signal = f'hcpya/{sub}/T1w/processed/fodf.nii.gz'
    else:
        signal = f'TractoInferno-ds003900/derivatives/{split}/{sub}/fodf/{sub}__fodf.nii.gz'
        
    signal = nib.load(signal)
    if not np.allclose(np.mean(signal.header.get_zooms()[:3]),signal.header.get_zooms()[0], atol=1e-03):
        print('WARNING: ODF SH file is not isotropic. Tracking cannot be '
                    'ran robustly. You are entering undefined behavior '
                    'territory.')
        
    data = set_sh_order_basis(signal.get_fdata(dtype=np.float32),
                                    'descoteaux07',
                                    target_order=8, #variant.get('target_sh_order', 8)
                                    target_basis='descoteaux07')

    if is_ismrm:
        tracking_approximated_mask = nib.load(f'atlas_dilated_masks/ismrm/{bundle}_5mm.nii.gz')
    elif not is_hcp:
        if is_ttoi_new:
            tracking_approximated_mask = nib.load(f'atlas_dilated_masks/ttoi/{sub}/{bundle_new}_5mm.nii.gz')
            ground_truth_mask = nib.load(f'TractoInferno-ds003900/Masks/{bundle_new}/{sub}__{bundle_new}.nii.gz')
        else:
            tracking_approximated_mask = nib.load(f'atlas_dilated_masks/ttoi/{sub}/{bundle}_5mm.nii.gz')
            ground_truth_mask = nib.load(f'TractoInferno-ds003900/Masks/{bundle}/{sub}__{bundle}.nii.gz')
    else:
        if is_hcp_new:
            tracking_approximated_mask = nib.load(f'105HCP/{split}/{sub}/GT_dilated_masks/3mm/{sub}_dilated_{bundle_new}.nii.gz')
        else:
            tracking_approximated_mask = nib.load(f'atlas_dilated_masks/hcp/{sub}/{bundle}_5mm.nii.gz')

    tracking_approximated_mask_data = tracking_approximated_mask.get_fdata()

    if len(tracking_approximated_mask.shape) == 3:
        tracking_approximated_mask_data = tracking_approximated_mask_data[..., None]

    signal_data = np.concatenate([data, tracking_approximated_mask_data], axis=-1) #CONCATENATED MASK WITH SIGNAL DATA
    signal_data = nib.Nifti1Image(signal_data, affine=signal.affine)

    gen_mask = generate_new_mask(model, signal_data, tracking_approximated_mask)

    if not is_hcp and not is_ismrm:
        print('Voxel wise metric before mask-refinement:')
        print(compute_voxel_pairwise_measures_4masks(tracking_approximated_mask.get_fdata(), ground_truth_mask.get_fdata()))
        print('Voxel wise metric after mask-refinement:')
        print(compute_voxel_pairwise_measures_4masks(gen_mask.get_fdata(), ground_truth_mask.get_fdata()))
    else:
        print('Mask refined and 1mm dilated -_-')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str)
    parser.add_argument('--bundle', type=str)
    parser.add_argument('--bundle_new', type=str)
    parser.add_argument('--saved_model_path', type=str)
    parser.add_argument('--hcp', type=bool, default=False) #add condition later
    parser.add_argument('--hcp_new', type=bool, default=False) #add condition later
    parser.add_argument('--ttoi_new', type=bool, default=False) #add condition later
    parser.add_argument('--ismrm', type=bool, default=False) #add condition later
    
    
    args = parser.parse_args()

    validation(variant=vars(args))

if __name__ == '__main__':
    main()