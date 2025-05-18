from dipy.data import get_sphere
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.reconst.csdeconv import sph_harm_ind_list
from scilpy.reconst.multi_processes import convert_sh_basis

import nibabel as nib
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import itertools

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mrm(variant):

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



    def get_neighborhood_features(data, tracking_approximated_mask_data, ground_truth_mask_data):
        # Define offsets for 6 neighbors (up, down, left, right, front, back)
        offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
        neighborhood_features = []
        labels = []

        # Iterate over each voxel
        for i, j, k in itertools.product(range(data.shape[0]), range(data.shape[1]), range(data.shape[2])):
            # Check if the voxel is within the tracking_approximated_mask
            if tracking_approximated_mask_data[i, j, k] == 1:
                features = []

                # Extract features for the current voxel
                features.extend(data[i, j, k, :])

                # Extract features for each neighbor
                for offset in offsets:
                    ni, nj, nk = i + offset[0], j + offset[1], k + offset[2]

                    # Check if the neighbor is within the bounds of the data
                    if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1] and 0 <= nk < data.shape[2]:
                        features.extend(data[ni, nj, nk, :])
                    else:
                        # If the neighbor is out of bounds, pad with zeros
                        features.extend([0] * data.shape[-1])

                neighborhood_features.append(features)
                labels.append(ground_truth_mask_data[i, j, k])

        return np.array(neighborhood_features), np.array(labels)



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

    subs_list = variant.get('subs_list')
    bundle = variant.get('bundle')

    save_models_path = f'mask_refinement_module/saved_models/{bundle}/'

    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)

    ttoi_path = 'TractoInferno-ds003900/derivatives/'
    splits = ['trainset', 'validset', 'testset']
    all_subs_dict = {split: get_subject_ids(ttoi_path, split) for split in splits}


    #data generation
    global_features = None  # Initialize as None
    global_gt_mask_labels = None
    for sub in subs_list:
        split = find_key_by_element(all_subs_dict, sub)
        print('Feature-gen for subject : ', sub, split)

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
    
        tracking_approximated_mask = nib.load(f'atlas_dilated_masks/ttoi/{sub}/{bundle}_5mm.nii.gz') # RBX atlas dilated mask for tract
        ground_truth_mask = nib.load(f'TractoInferno-ds003900/Masks/{bundle}/{sub}__{bundle}.nii.gz') # Binary mask of Groundtruth tract

        tracking_approximated_mask_data = tracking_approximated_mask.get_fdata()

        print('Pop-averaged ki shape: ', tracking_approximated_mask_data.shape)

        if len(tracking_approximated_mask.shape) == 3:
            tracking_approximated_mask_data = tracking_approximated_mask_data[..., None]

        signal_data = np.concatenate([data, tracking_approximated_mask_data], axis=-1) #CONCATENATED MASK WITH SIGNAL DATA
        signal_data = nib.Nifti1Image(signal_data, affine=signal.affine)

        print('Signal ki shape: ', signal_data.shape)

        sub_features, gt_mask_labels = get_neighborhood_features(signal_data.get_fdata(), tracking_approximated_mask_data, ground_truth_mask.get_fdata())
        print("Shape of neighborhood features:", sub_features.shape)
        print("Shape of gt_mask_labels:", gt_mask_labels.shape)

        # Concatenate sub_features to global_features
        if global_features is None:
            global_features = sub_features
            global_gt_mask_labels = gt_mask_labels
        else:
            global_features = np.concatenate((global_features, sub_features), axis=0)
            global_gt_mask_labels = np.concatenate((global_gt_mask_labels, gt_mask_labels), axis=0)

    #number of total global features to train on
    print("Total data-samples: ", global_features.shape[0])

    # Convert NumPy array-data to PyTorch tensors
    global_features_tensor = torch.tensor(global_features, dtype=torch.float32).to('cuda')
    global_gt_mask_labels_tensor = torch.tensor(global_gt_mask_labels, dtype=torch.float32).to('cuda')

    # Define dataset and dataloader
    dataset = TensorDataset(global_features_tensor, global_gt_mask_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Define the model and move it to CUDA
    model = SimpleNN().to('cuda')

    if(variant.get('load_model')):
        # Load the saved model state dictionary
        saved_model_path = variant.get('saved_model_path')
        model.load_state_dict(torch.load(saved_model_path))
        print(f'Loaded model {saved_model_path}')

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Lists to store loss values and epochs for plotting
    loss_values = []
    epoch_values = []

    # Training loop
    num_epochs = 101
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Save model after every 10th epoch starting from 20th epoch
        if epoch >= 10 and (epoch - 10) % 10 == 0:
            torch.save(model.state_dict(), f'{save_models_path}model_epoch_{epoch}.pt')
        
        # Show loss after every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
        
        # Append loss and epoch values for plotting
        loss_values.append(running_loss / len(dataloader))
        epoch_values.append(epoch + 1)

    # Plot loss values
    plt.plot(epoch_values, loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'loss{bundle}_plot.png')
    plt.show()

    print("Trained")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subs_list', nargs='+')#, required=True)
    parser.add_argument('--bundle', type=str)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--saved_model_path', type=str)
    
    args = parser.parse_args()

    mrm(variant=vars(args))

if __name__ == '__main__':
    main()