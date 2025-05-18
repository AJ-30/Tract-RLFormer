import nibabel as nib
from scipy import ndimage
import sys
import os

if len(sys.argv) != 4:
    print("Usage: python script.py input_file output_file radius")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
radius = int(sys.argv[3])

try:
    cc_gen_mask = nib.load(input_file)
except FileNotFoundError:
    print("Input file not found.")
    sys.exit(1)

mask_data = cc_gen_mask.get_fdata()
dilated_mask = ndimage.binary_dilation(mask_data, iterations=radius).astype(mask_data.dtype)
dilated_mask_nifti = nib.Nifti1Image(dilated_mask, cc_gen_mask.affine)

output_folder = os.path.dirname(output_file)
try:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    nib.save(dilated_mask_nifti, output_file)
except Exception as e:
    print("Error occurred while saving the output file:", e)
    sys.exit(1)

print("Dilated mask saved successfully to", output_file)
