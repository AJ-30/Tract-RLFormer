import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import nibabel as nib
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from nibabel.streamlines import Tractogram
from enum import Enum
import torch 
import functools
from gymnasium.wrappers.normalize import RunningMeanStd
from typing import Callable, Dict, Tuple



def convert_length_mm2vox(
    length_mm: float, affine: np.ndarray
) -> float:
    """Convert a length from mm to voxel space (if the space is isometric).

    Parameters
    ----------
    length_mm : float
        Length in mm.
    affine : np.ndarray
        Affine to bring coordinates from voxel space to rasmm space, usually
        provided with an anatomy file.

    Returns
    -------
    length_vox : float
        Length expressed in isometric voxel units.

    Raises
    ------
    ValueError
        If the voxel space is not isometric
        (different absolute values on the affine diagonal).
    """
    diag = np.diagonal(affine)[:3]
    vox2mm = np.mean(np.abs(diag))

    # Affine diagonal should have the same absolute value
    # for an isometric space
    if not np.allclose(np.abs(diag), vox2mm, rtol=5e-2, atol=5e-2):
        raise ValueError("Voxel space is not iso, "
                         " cannot convert a scalar length "
                         "in mm to voxel space. "
                         "Affine provided : {}".format(affine))

    length_vox = length_mm / vox2mm
    return length_vox

def interpolate_volume_at_coordinates(
    volume: np.ndarray,
    coords: np.ndarray,
    mode: str = 'nearest',
    order: int = 1,
    cval: float = 0.0
) -> np.ndarray:
    """ Evaluates a 3D or 4D volume data at the given coordinates using
    trilinear interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’).
        Default is ‘nearest’.
        ('constant' uses 0.0 as a points outside the boundary)

    Returns
    -------
    output : 2D array
        Values from volume.
    """
    # map_coordinates uses the center of the voxel, so should we shift to
    # the corner?

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(
            volume, coords.T, order=order, mode=mode, cval=cval)

    if volume.ndim == 4:
        D = volume.shape[-1]
        values_4d = np.zeros((coords.shape[0], D))
        for i in range(volume.shape[-1]):
            values_4d[:, i] = map_coordinates(
                volume[..., i], coords.T, order=order,
                mode=mode, cval=cval)
        return values_4d
    
def is_inside_mask(
    streamlines: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.
):
    """ Checks which streamlines have their last coordinates inside a mask.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    mask : 3D `numpy.ndarray`
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    inside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline's last coordinate is inside the mask
        or not.
    """
    # Get last streamlines coordinates
    return interpolate_volume_at_coordinates(
        mask, streamlines[:, -1, :], mode='constant', order=0) >= threshold


def normalize_vectors(v):
    return v / np.sqrt(np.sum(v ** 2, axis=-1, keepdims=True))

def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """ Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
        each other.

    Returns
    -------
    too_curvy : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too curvy or not
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian
    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        # return np.zeros(streamlines.shape[0], dtype=np.uint8) #COMMENTED BY ASHUTOSH
        return np.full(streamlines.shape[0], False)#ADDED BY ASHUTOSH

    # Compute vectors for the last and before last streamline segments
    u = normalize_vectors(streamlines[:, -1] - streamlines[:, -2])
    v = normalize_vectors(streamlines[:, -2] - streamlines[:, -3])

    # Compute angles
    angles = np.arccos(np.sum(u * v, axis=1).clip(-1., 1.))
    # print('----------------------------')
    # print('----------------------------')
    # print(np.sum(u * v, axis=1))
    # print('----------------------------')
    # print(angles)
    # print('----------------------------')
    # print('----------------------------')

    return angles > max_theta_rad

def is_too_long(streamlines: np.ndarray, max_nb_steps: int):
    """ Checks whether streamlines have exceeded the maximum number of steps

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    max_nb_steps : int
        Maximum number of steps a streamline can have

    Returns
    -------
    too_long : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array telling whether a streamline is too long or not
    """
    return np.full(streamlines.shape[0], streamlines.shape[1] >= max_nb_steps)

class Reward(object):

    def __init__(
        self,
        peaks: nib.nifti1.Nifti1Image = None,
        exclude: nib.nifti1.Nifti1Image = None,
        target: nib.nifti1.Nifti1Image = None,
        max_nb_steps: float = 200,
        theta: float = 60,
        min_nb_steps: float = 10,
        asymmetric: bool = False,
        alignment_weighting: float = 1.0,
        straightness_weighting: float = 0.0,
        length_weighting: float = 0.0,
        target_bonus_factor: float = 0.0,
        exclude_penalty_factor: float = 0.0,
        angle_penalty_factor: float = 0.0,
        scoring_data: str = None,
        reference: str = None
    ):
        """
        peaks: `MRIDataVolume`
            Volume containing the fODFs peaks
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones
        max_len: `float`
            Maximum lengths for the streamlines (in terms of points)
        theta: `float`
            Maximum degrees between two streamline segments
        alignment_weighting: `float`
            Coefficient for how much reward to give to the alignment
            with peaks
        straightness_weighting: `float`
            Coefficient for how much reward to give to the alignment
            with the last streamline segment
        length_weighting: `float`
            Coefficient for how much to reward the streamline for being
            long
        target_bonus_factor: `float`
            Bonus for streamlines reaching the target mask
        exclude_penalty_factor: `float`
            Penalty for streamlines reaching the exclusion mask
        angle_penalty_factor: `float`
            Penalty for looping or too-curvy streamlines
        """

        print('Initializing reward with factors')
        print({'alignment': alignment_weighting,
               'straightness': straightness_weighting,
               'length': length_weighting,
               'target': target_bonus_factor,
               'exclude_penalty_factor': exclude_penalty_factor,
               'angle_penalty_factor': angle_penalty_factor})

        self.peaks = peaks
        self.exclude = exclude
        self.target = target
        self.max_nb_steps = max_nb_steps
        self.theta = theta
        self.min_nb_steps = min_nb_steps
        self.asymmetric = asymmetric
        self.alignment_weighting = alignment_weighting
        self.straightness_weighting = straightness_weighting
        self.length_weighting = length_weighting
        self.target_bonus_factor = target_bonus_factor
        self.exclude_penalty_factor = exclude_penalty_factor
        self.angle_penalty_factor = angle_penalty_factor
        self.scoring_data = scoring_data
        self.reference = reference

        # if self.scoring_data:
        #     print('WARNING: Rewarding from the Tractometer is not currently '
        #           'officially supported and may not work. If you do want to '
        #           'improve Track-to-Learn and make it work, I can happily '
        #           'help !')

        #     gt_bundles_attribs_path = os.path.join(
        #         self.scoring_data,
        #         'gt_bundles_attributes.json')

        #     basic_bundles_attribs = load_attribs(gt_bundles_attribs_path)

        #     # Prepare needed scoring data
        #     masks_dir = os.path.join(self.scoring_data, "masks")
        #     rois_dir = os.path.join(masks_dir, "rois")
        #     bundles_dir = os.path.join(self.scoring_data, "bundles")
        #     bundles_masks_dir = os.path.join(masks_dir, "bundles")
        #     ref_anat_fname = os.path.join(masks_dir, "wm.nii.gz")

        #     ROIs = [nib.load(os.path.join(rois_dir, f))
        #             for f in sorted(os.listdir(rois_dir))]

        #     # Get the dict with 'name', 'threshold', 'streamlines',
        #     # 'cluster_map' and 'mask' for each bundle.
        #     ref_bundles = _prepare_gt_bundles_info(bundles_dir,
        #                                            bundles_masks_dir,
        #                                            basic_bundles_attribs,
        #                                            ref_anat_fname)

        #     self.scoring_function = functools.partial(
        #         score,
        #         ref_bundles=ref_bundles,
        #         ROIs=ROIs,
        #         compute_ic_ib=False)

    def __call__(self, streamlines, dones):
        """
        Compute rewards for the last step of the streamlines
        Each reward component is weighted according to a
        coefficient

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space

        Returns
        -------
        rewards: `float`
            Reward components weighted by their coefficients as well
            as the penalites
        """

        N = len(streamlines)

        length = reward_length(streamlines, self.max_nb_steps) \
            if self.length_weighting > 0. else np.zeros((N), dtype=np.uint8)
        #ASHUTOSH------- required if tracking via ttl_track.py
        # self.peaks = get_peaks()
        #------------
        alignment = reward_alignment_with_peaks(
            streamlines, self.peaks.get_fdata(), self.asymmetric) \
            if self.alignment_weighting > 0 else np.zeros((N), dtype=np.uint8)
        straightness = reward_straightness(streamlines) \
            if self.straightness_weighting > 0 else \
            np.zeros((N), dtype=np.uint8)

        weights = np.asarray([
            self.alignment_weighting, self.straightness_weighting,
            self.length_weighting])
        params = np.stack((alignment, straightness, length))
        rewards = np.dot(params.T, weights)

        # Penalize sharp turns
        if self.angle_penalty_factor > 0.:
            rewards += penalize_sharp_turns(
                streamlines, self.theta, self.angle_penalty_factor)

        # Penalize streamlines ending in exclusion mask
        if self.exclude_penalty_factor > 0.:
            rewards += penalize_exclude(
                streamlines,
                self.exclude.get_fdata(),
                self.exclude_penalty_factor)

        # Reward streamlines ending in target mask
        if self.target_bonus_factor > 0.:
            rewards += self.reward_target(
                streamlines,
                dones)

        return rewards

    def reward_target(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):
        """ Reward streamlines if they end up in the GM

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        target: np.ndarray
            Grey matter mask
        penalty_factor: `float`
            Penalty for streamlines ending in target mask
            Should be >= 0

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """
        target_streamlines = is_inside_mask(
            streamlines, self.target.get_fdata(), 0.5
        ) * self.target_bonus_factor

        reward = target_streamlines * dones * int(
            streamlines.shape[1] > self.min_nb_steps)

        return reward

    def reward_tractometer(
        self,
        streamlines: np.ndarray,
        dones: np.ndarray,
    ):
        """ Reward streamlines if the Tractometer marks them as valid.

        **WARNING**: This function is not supported and may not work. I
        wrote it as part of some experimentation and I forgot to remove it
        when releasing the code. Let me know if you want help making this
        work.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        target: np.ndarray
            Grey matter mask
        penalty_factor: `float`
            Penalty for streamlines ending in target mask
            Should be >= 0

        Returns
        -------
        rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array containing the reward
        """

        # Get boolean array of streamlines ending in mask * penalty
        if streamlines.shape[1] >= self.min_nb_steps and np.any(dones):
            # Should the SFT be moved to RASMM space for scoring ? To corner
            # or to center ?
            sft = StatefulTractogram(streamlines, self.reference, Space.VOX)
            to_score = np.arange(len(sft))[dones]
            sub_sft = sft[to_score]
            VC, IC, NC = self.scoring_function(sub_sft)

            # The factors for positively and negatively rewarding streamlines
            # as well as which to apply positive, negative or no reward is 
            # open for improvements. I have not thoroughly tested anything.

            reward = np.zeros((streamlines.shape[0]))
            if len(VC) > 0:
                reward[to_score[VC]] += self.target_bonus_factor
                # Display which streamlines are positively rewarded
                # self.render(self.peaks, streamlines[to_score[VC]],
                #             reward[to_score[VC]])
            if len(IC) > 0:
                reward[to_score[IC]] -= self.target_bonus_factor
            if len(NC) > 0:
                reward[to_score[NC]] -= self.target_bonus_factor
        else:
            reward = np.zeros((streamlines.shape[0]))
        return reward

    def render(
        self,
        peaks,
        streamlines,
        rewards
    ):
        """ Debug function

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        tractogram = Tractogram(
            streamlines=streamlines,
            data_per_streamline={
                'seeds': streamlines[:, 0, :]
            })

        # Reshape peaks for displaying
        X, Y, Z, M = peaks.get_fdata().shape
        peaks = np.reshape(peaks.get_fdata(), (X, Y, Z, 5, M//5))

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines, rewards)
        peak_actor = actor.peak_slicer(peaks,
                                       np.ones((X, Y, Z, M)),
                                       colors=(0.2, 0.2, 1.),
                                       opacity=0.5)
        mask_actor = actor.contour_from_roi(
            self.target.get_fdata())

        dot_actor = actor.dots(tractogram.data_per_streamline['seeds'],
                               color=(1, 1, 1),
                               opacity=1,
                               dot_size=2.5)
        scene.add(stream_actor)
        scene.add(peak_actor)
        scene.add(dot_actor)
        scene.add(mask_actor)
        scene.reset_camera_tight(0.95)

        showm = window.ShowManager(scene, reset_camera=True)
        showm.initialize()
        showm.start()


def penalize_exclude(streamlines, exclude, penalty_factor):
    """ Penalize streamlines if they loop

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    exclude: np.ndarray
        CSF matter mask
    penalty_factor: `float`
        Penalty for streamlines ending in exclusion mask
        Should be <= 0

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return \
        is_inside_mask(
            streamlines, exclude, 0.5) * -penalty_factor


def penalize_sharp_turns(streamlines, theta, penalty_factor):
    """ Penalize streamlines if they curve too much

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space
    theta: `float`
        Maximum angle between streamline steps
    penalty_factor: `float`
        Penalty for looping or too-curvy streamlines

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    return is_too_curvy(streamlines, theta) * -penalty_factor


def reward_length(streamlines, max_length):
    """ Reward streamlines according to their length w.r.t the maximum length

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, S, _ = streamlines.shape

    rewards = np.asarray([S] * N) / max_length

    return rewards


def reward_alignment_with_peaks(
    streamlines, peaks, asymmetric
):
    """ Reward streamlines according to the alignment to their corresponding
        fODFs peaks

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the reward
    """
    N, L, _ = streamlines.shape

    if streamlines.shape[1] < 2:
        # Not enough segments to compute curvature
        return np.ones(len(streamlines), dtype=np.uint8)

    X, Y, Z, P = peaks.shape
    idx = streamlines[:, -2].astype(np.int32)

    # Get peaks at streamline end
    v = interpolate_volume_at_coordinates(
        peaks, idx, mode='nearest', order=0)

    # Presume 5 peaks (per hemisphere if asymmetric)
    if asymmetric:
        v = np.reshape(v, (N, 5 * 2, P // (5 * 2)))
    else:
        v = np.reshape(v, (N, 5, P // 5))

        with np.errstate(divide='ignore', invalid='ignore'):
            # # Normalize peaks
            v = normalize_vectors(v)

        # Zero NaNs
        v = np.nan_to_num(v)

    # Get last streamline segments

    dirs = np.diff(streamlines, axis=1)
    u = dirs[:, -1]
    # Normalize segments
    with np.errstate(divide='ignore', invalid='ignore'):
        u = normalize_vectors(u)

    # Zero NaNs
    u = np.nan_to_num(u)

    # Get do product between all peaks and last streamline segments
    dot = np.einsum('ijk,ik->ij', v, u)

    if not asymmetric:
        dot = np.abs(dot)

    # Get alignment with the most aligned peak
    rewards = np.amax(dot, axis=-1)
    # rewards = np.abs(dot)

    factors = np.ones((N))

    # Weight alignment with peaks with alignment to itself
    if streamlines.shape[1] >= 3:
        # Get previous to last segment
        w = dirs[:, -2]

        # # Normalize segments
        with np.errstate(divide='ignore', invalid='ignore'):
            w = normalize_vectors(w)

        # # Zero NaNs
        w = np.nan_to_num(w)

        # Calculate alignment between two segments
        np.einsum('ik,ik->i', u, w, out=factors)

    # Penalize angle with last step
    rewards *= factors

    return rewards


def reward_straightness(streamlines):
    """ Reward streamlines according to its sinuosity

    Distance between start and end of streamline / length

    A perfectly straight line has 1.
    A circle would have 0.

    Parameters
    ----------
    streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space

    Returns
    -------
    rewards: 1D boolean `numpy.ndarray` of shape (n_streamlines,)
        Array containing the angle between the last two segments
e   """

    N, S, _ = streamlines.shape

    start = streamlines[:, 0]
    end = streamlines[:, -1]

    step_size = 1.
    reward = np.linalg.norm(end - start, axis=1) / (S * step_size)

    return np.clip(reward + 0.5, 0, 1)


# Flags enum
class StoppingFlags(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop
    """
    STOPPING_MASK = int('00000001', 2)
    STOPPING_LENGTH = int('00000010', 2)
    STOPPING_CURVATURE = int('00000100', 2)
    STOPPING_TARGET = int('00001000', 2)
    STOPPING_LOOP = int('00010000', 2)

class CmcStoppingCriterion(object):
    """ Checks which streamlines should stop according to Continuous map
    criteria.
    Ref:
        Girard, G., Whittingstall, K., Deriche, R., & Descoteaux, M. (2014)
        Towards quantitative connectivity analysis: reducing tractography
        biases.
        Neuroimage, 98, 266-278.

    This is only in the partial-spirit of CMC. A good improvement (#TODO)
    would be to include or exclude streamlines from the resulting
    tractogram as well. Let me know if you need help in adding this
    functionnality.
    """

    def __init__(
        self,
        include_mask: np.ndarray,
        exclude_mask: np.ndarray,
        affine: np.ndarray,
        step_size: float,
        min_nb_steps: int,
    ):
        """
        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            3D image defining a stopping mask. The interior of the mask is
            defined by values higher or equal than `threshold` .
        affine: `numpy.ndarray` with shape (4,4) (optional)
            Tranformation that aligns brings streamlines to rasmm from vox.
        threshold : float
            Voxels with a value higher or equal than this threshold are
            considered as part of the interior of the mask.
        """
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.affine = affine
        vox_size = np.mean(np.abs(np.diag(affine)[:3]))
        self.correction_factor = step_size / vox_size
        self.min_nb_steps = min_nb_steps

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """
        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        Returns
        -------
        outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array telling whether a streamline's last coordinate is outside the
            mask or not.
        """

        include_result = interpolate_volume_at_coordinates(
            self.include_mask, streamlines[:, -1, :], mode='constant',
            order=1)
        if streamlines.shape[1] < self.min_nb_steps:
            include_result[:] = 0.

        exclude_result = interpolate_volume_at_coordinates(
            self.exclude_mask, streamlines[:, -1, :], mode='constant',
            order=1, cval=1.0)

        # If streamlines are still in 100% WM, don't exit
        wm_points = include_result + exclude_result <= 0

        # Compute continue probability
        num = np.maximum(0, (1 - include_result - exclude_result))
        den = num + include_result + exclude_result
        p = (num / den) ** self.correction_factor

        # p >= continue prob -> not continue
        not_continue_points = np.random.random(streamlines.shape[0]) >= p

        # if by some magic some wm point don't continue, make them continue
        not_continue_points[wm_points] = False

        # if the point is in the include map, it has potentially reached GM
        p = (include_result / (include_result + exclude_result))
        stop_include = np.random.random(streamlines.shape[0]) < p
        not_continue_points[stop_include] = True

        return not_continue_points
    
class BinaryStoppingCriterion(object):
    """
    Defines if a streamline is outside a mask using NN interp.
    """

    def __init__(
        self,
        mask: np.ndarray,
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            3D image defining a stopping mask. The interior of the mask is
            defined by values higher or equal than `threshold` .
        threshold : float
            Voxels with a value higher or equal than this threshold are
            considered as part of the interior of the mask.
        """
        self.mask = mask
        self.threshold = threshold

    def __call__(
        self,
        streamlines: np.ndarray,
    ):
        """ Checks which streamlines have their last coordinates outside a
        mask.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        Returns
        -------
        outside : 1D boolean `numpy.ndarray` of shape (n_streamlines,)
            Array telling whether a streamline's last coordinate is outside the
            mask or not.
        """

        # Get last streamlines coordinates
        return interpolate_volume_at_coordinates(
            self.mask, streamlines[:, -1, :], mode='constant',
            order=0) < self.threshold
    

def get_neighborhood_directions(
    radius: float
) -> np.ndarray:
    """ Returns predefined neighborhood directions at exactly `radius` length
        For now: Use the 6 main axes as neighbors directions, plus (0,0,0)
        to keep current position

    Parameters
    ----------
    radius : float
        Distance to neighbors

    Returns
    -------
    directions : `numpy.ndarray` with shape (n_directions, 3)

    Notes
    -----
    Coordinates are in voxel-space
    """
    axes = np.identity(3)
    directions = np.concatenate(([[0, 0, 0]], axes, -axes)) * radius
    return directions



def torch_trilinear_interpolation(
    volume: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords : torch.Tensor with shape (N,3)
        The coordinates where to interpolate

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    # Get device, and make sure volume and coords are using the same one
    assert volume.device == coords.device, "volume on device: {}; " \
                                           "coords on device: {}".format(
                                               volume.device,
                                               coords.device)
    coords = coords.type(torch.float32)
    volume = volume.type(torch.float32)

    device = volume.device

    B1_torch = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 1, 0, 0, 0],
                             [-1, 0, 1, 0, 0, 0, 0, 0],
                             [-1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, -1, 0, -1, 0, 1, 0],
                             [1, -1, -1, 1, 0, 0, 0, 0],
                             [1, -1, 0, 0, -1, 1, 0, 0],
                             [-1, 1, 1, -1, 1, -1, -1, 1]],
                            dtype=torch.float32, device=device)

    idx_torch = torch.tensor([[0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 0],
                              [1, 0, 1],
                              [1, 1, 0],
                              [1, 1, 1]], dtype=torch.float32, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.dim() == 3:
        # torch needs indices to be cast to long
        indices_unclipped = (
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0]).to(device)
        upper = (torch.as_tensor(volume.shape) - 1).to(device)
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[
            indices[:, 0], indices[:, 1], indices[:, 2]
        ].reshape((coords.shape[0], -1)).t()

        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy, dy * dz,
            dx * dz, dx * dy * dz],
            dim=0)
        output = torch.sum(P * torch.mm(B1_torch.t(), Q1), dim=0)

        return output

    if volume.dim() == 4:
        # 8 coordinates of the corners of the cube, for each input coordinate
        indices_unclipped = torch.floor(
            coords[:, None, :] + idx_torch).reshape((-1, 3)).long()

        # Clip indices to make sure we don't go out-of-bounds
        lower = torch.as_tensor([0, 0, 0], device=device)
        upper = torch.as_tensor(volume.shape[:3], device=device) - 1
        indices = torch.min(torch.max(indices_unclipped, lower), upper)

        # Fetch volume data at indices
        P = volume[indices[:, 0], indices[:, 1], indices[:, 2], :].reshape(
            (coords.shape[0], 8, volume.shape[-1]))

        # Shift 0.5 because fODFs are centered ?
        # coords = coords - 0.5
        d = coords - torch.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = torch.stack([
            torch.ones_like(dx), dx, dy, dz, dx * dy,
            dy * dz, dx * dz, dx * dy * dz],
            dim=0)
        output = torch.sum(
            P * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output.type(torch.float32)

    raise ValueError(
        "There was a problem with the volume's number of dimensions!")



def get_sh(
    segments,
    data_volume,
    add_neighborhood_vox,
    neighborhood_directions,
    history,
    device
) -> np.ndarray:
    """ Get the sh coefficients at the end of streamlines
    """

    N, H, P = segments.shape
    flat_coords = np.reshape(segments, (N * H, P))

    coords = torch.as_tensor(flat_coords).to(device)
    n_coords = coords.shape[0]

    if add_neighborhood_vox:
        # Extend the coords array with the neighborhood coordinates
        coords = torch.repeat_interleave(
            coords,
            neighborhood_directions.size()[0],
            axis=0)

        coords[:, :3] += \
            neighborhood_directions.repeat(n_coords, 1)

        # Evaluate signal as if all coords were independent
        partial_signal = torch_trilinear_interpolation(
            data_volume, coords)

        # Reshape signal into (n_coords, new_feature_size)
        new_feature_size = partial_signal.size()[-1] * \
            neighborhood_directions.size()[0]
    else:
        partial_signal = torch_trilinear_interpolation(
            data_volume,
            coords).type(torch.float32)
        new_feature_size = partial_signal.size()[-1]

    signal = torch.reshape(partial_signal, (N, history * new_feature_size))

    assert len(signal.size()) == 2, signal.size()

    return signal


# CHANGED BaseENV, always TODO: 

class BaseEnv(object):
    """
    Abstract tracking environment.
    TODO: Add more explanations
    """

    def __init__(
        self,
        input_volume: nib.nifti1.Nifti1Image,
        tracking_mask: nib.nifti1.Nifti1Image,
        target_mask: nib.nifti1.Nifti1Image,
        seeding_mask: nib.nifti1.Nifti1Image,
        peaks: nib.nifti1.Nifti1Image,
        env_dto: dict,
        include_mask: nib.nifti1.Nifti1Image = None,
        exclude_mask: nib.nifti1.Nifti1Image = None,
    ):
        """
        Parameters
        ----------
        input_volume: MRIDataVolume
            Volumetric data containing the SH coefficients
        tracking_mask: MRIDataVolume
            Volumetric mask where tracking is allowed
        target_mask: MRIDataVolume
            Mask representing the tracking endpoints
        seeding_mask: MRIDataVolume
            Mask where seeding should be done
        peaks: MRIDataVolume
            Volume containing the fODFs peaks
        env_dto: dict
            DTO containing env. parameters
        include_mask: MRIDataVolume
            Mask representing the tracking go zones. Only useful if
            using CMC.
        exclude_mask: MRIDataVolume
            Mask representing the tracking no-go zones. Only useful if
            using CMC.
        """

        # Volumes and masks
        self.affine = input_volume.affine
        self.affine_rasmm2vox = np.linalg.inv(self.affine)

        self.data_volume = torch.tensor(
            input_volume.get_fdata(), dtype=torch.float32, device=env_dto['device'])
        self.tracking_mask = tracking_mask
        self.target_mask = target_mask
        self.include_mask = include_mask
        self.exclude_mask = exclude_mask
        self.peaks = peaks

        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        self.reference = env_dto['reference']

        # Tracking parameters
        self.n_signal = env_dto['n_signal']
        self.n_dirs = env_dto['n_dirs']
        self.theta = theta = env_dto['theta']
        self.npv = env_dto['npv']
        self.cmc = env_dto['cmc']
        self.asymmetric = env_dto['asymmetric']

        step_size_mm = env_dto['step_size']
        min_length_mm = env_dto['min_length']
        max_length_mm = env_dto['max_length']
        add_neighborhood_mm = env_dto['add_neighborhood']

        # Reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        self.straightness_weighting = env_dto['straightness_weighting']
        self.length_weighting = env_dto['length_weighting']
        self.target_bonus_factor = env_dto['target_bonus_factor']
        self.exclude_penalty_factor = env_dto['exclude_penalty_factor']
        self.angle_penalty_factor = env_dto['angle_penalty_factor']
        self.compute_reward = env_dto['compute_reward']
        self.scoring_data = env_dto['scoring_data']

        self.rng = env_dto['rng']
        self.device = env_dto['device']

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not
        self.stopping_criteria = {}
        mask_data = tracking_mask.get_fdata().astype(np.uint8)

        self.seeding_data = seeding_mask.get_fdata().astype(np.uint8)

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine)
        self.min_length = min_length_mm
        self.max_length = max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        if self.compute_reward:
            self.reward_function = Reward(
                peaks=self.peaks,
                exclude=self.exclude_mask,
                target=self.target_mask,
                max_nb_steps=self.max_nb_steps,
                theta=self.theta,
                min_nb_steps=self.min_nb_steps,
                asymmetric=self.asymmetric,
                alignment_weighting=self.alignment_weighting,
                straightness_weighting=self.straightness_weighting,
                length_weighting=self.length_weighting,
                target_bonus_factor=self.target_bonus_factor,
                exclude_penalty_factor=self.exclude_penalty_factor,
                angle_penalty_factor=self.angle_penalty_factor,
                scoring_data=self.scoring_data,
                reference=self.reference)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy, max_theta=theta)

        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.get_fdata(),
                self.exclude_mask.get_fdata(),
                self.affine,
                self.step_size,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion
        else:
            binary_criterion = BinaryStoppingCriterion(
                mask_data,
                0.5)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
                binary_criterion

        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_LOOP] = \
        #     functools.partial(is_looping,
        #                       loop_threshold=300)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood_mm:
            self.add_neighborhood_vox = convert_length_mm2vox(
                add_neighborhood_mm,
                self.affine)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(self.device)

        # Tracking seeds
        self.seeds = self._get_tracking_seeds_from_mask(
            self.seeding_data,
            self.npv,
            self.rng)
        print(
            '{} has {} seeds.'.format(self.__class__.__name__,
                                      len(self.seeds)))
        
    def get_state_size(self):
        example_state = self.reset(0, 1)
        self._state_size = example_state.shape[1]
        return self._state_size

    def get_action_size(self):
        """ TODO: Support spherical actions"""

        return 3

    def get_voxel_size(self):
        """ Returns the voxel size by taking the mean value of the diagonal
        of the affine. This implies that the vox size is always isometric.

        Returns
        -------
        voxel_size: float
            Voxel size in mm.

        """
        diag = np.diagonal(self.affine)[:3]
        voxel_size = np.mean(np.abs(diag))

        return voxel_size

    def set_step_size(self, step_size_mm):
        """ Set a different step size (in voxels) than computed by the
        environment. This is necessary when the voxel size between training
        and tracking envs is different.
        """

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine)

        if self.add_neighborhood_vox:
            self.add_neighborhood_vox = convert_length_mm2vox(
                step_size_mm,
                self.affine)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(self.device)

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        if self.compute_reward:
            self.reward_function = Reward(
                peaks=self.peaks,
                exclude=self.exclude_mask,
                target=self.target_mask,
                max_nb_steps=self.max_nb_steps,
                theta=self.theta,
                min_nb_steps=self.min_nb_steps,
                asymmetric=self.asymmetric,
                alignment_weighting=self.alignment_weighting,
                straightness_weighting=self.straightness_weighting,
                length_weighting=self.length_weighting,
                target_bonus_factor=self.target_bonus_factor,
                exclude_penalty_factor=self.exclude_penalty_factor,
                angle_penalty_factor=self.angle_penalty_factor,
                scoring_data=self.scoring_data,
                reference=self.reference)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.get_fdata(),
                self.exclude_mask.get_fdata(),
                self.affine,
                self.step_size,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion

    def _normalize(self, obs):
        """Normalises the observation using the running mean and variance of
        the observations. Taken from Gymnasium."""
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=(self._state_size,))
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)

    def _get_tracking_seeds_from_mask(
        self,
        mask: np.ndarray,
        npv: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """ Given a binary seeding mask, get seeds in DWI voxel
        space using the provided affine. TODO: Replace this
        with scilpy's SeedGenerator

        Parameters
        ----------
        mask : 3D `numpy.ndarray`
            Binary seeding mask
        npv : int
        rng : `numpy.random.RandomState`

        Returns
        -------
        seeds : `numpy.ndarray`
        """
        seeds = []
        indices = np.array(np.where(mask)).T
        for idx in indices:
            seeds_in_seeding_voxel = idx + rng.uniform(
                -0.5,
                0.5,
                size=(npv, 3))
            seeds.extend(seeds_in_seeding_voxel)
        seeds = np.array(seeds, dtype=np.float16)
        return seeds

    def _format_state(
        self,
        streamlines: np.ndarray
    ) -> np.ndarray:
        """
        From the last streamlines coordinates, extract the corresponding
        SH coefficients

        Parameters
        ----------
        streamlines: `numpy.ndarry`
            Streamlines from which to get the coordinates

        Returns
        -------
        inputs: `numpy.ndarray`
            Observations of the state, incl. previous directions.
        """
        N, L, P = streamlines.shape
        if N <= 0:
            return []
        segments = streamlines[:, -1, :][:, None, :]

        signal = get_sh(
            segments,
            self.data_volume,
            self.add_neighborhood_vox,
            self.neighborhood_directions,
            self.n_signal,
            self.device
        )

        N, S = signal.shape

        inputs = torch.zeros((N, S + (self.n_dirs * P)), device=self.device)

        inputs[:, :S] = signal

        previous_dirs = np.zeros((N, self.n_dirs, P), dtype=np.float32)
        if L > 1:
            dirs = np.diff(streamlines, axis=1)
            previous_dirs[:, :min(dirs.shape[1], self.n_dirs), :] = \
                dirs[:, :-(self.n_dirs+1):-1, :]

        dir_inputs = torch.reshape(
            torch.from_numpy(previous_dirs).to(self.device),
            (N, self.n_dirs * P))

        inputs[:, S:] = dir_inputs

        # if self.normalize_obs and self._state_size is not None:
        #     inputs = self._normalize(inputs)

        return inputs

    def _filter_stopping_streamlines(
        self,
        streamlines: np.ndarray,
        stopping_criteria: Dict[StoppingFlags, Callable]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Checks which streamlines should stop and which ones should
        continue.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamline coordinates in voxel space
        stopping_criteria : dict of int->Callable
            List of functions that take as input streamlines, and output a
            boolean numpy array indicating which streamlines should stop

        Returns
        -------
        should_stop : `numpy.ndarray`
            Boolean array, True is tracking should stop
        flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline
        """
        idx = np.arange(len(streamlines))

        should_stop = np.zeros(len(idx), dtype=np.bool_)
        flags = np.zeros(len(idx), dtype=int)

        # For each possible flag, determine which streamline should stop and
        # keep track of the triggered flag
        # print(type(stopping_criteria))
        for flag, stopping_criterion in stopping_criteria.items():
            stopped_by_criterion = stopping_criterion(streamlines)
            flags[stopped_by_criterion] |= flag.value
            # print("-----")
            # print(flag)
            # print(flag.value)
            # print(flags[stopped_by_criterion])
            # print(stopping_criterion)
            # print("-----")
            # # print("-----")
            # print("what is value of stopping criteria on streamline: ",stopping_criterion(streamlines))
            # if flag.value == 
            should_stop[stopped_by_criterion] = True#ASHUTOSH

        return should_stop, flags

    def _is_stopping():
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria
        """
        pass

    def reset():
        """ Initialize tracking seeds and streamlines
        """
        pass

    def step():
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done, and compute new
        hidden states
        """
        pass

    def render(
        self,
        tractogram: Tractogram = None,
        filename: str = None
    ):
        """ Render the streamlines, either directly or through a file
        Might render from "outside" the environment, like for comet

        Parameters:
        -----------
        tractogram: Tractogram, optional
            Object containing the streamlines and seeds
        path: str, optional
            If set, save the image at the specified location instead
            of displaying directly
        """
        from fury import window, actor
        # Might be rendering from outside the environment
        if tractogram is None:
            tractogram = Tractogram(
                streamlines=self.streamlines[:, :self.length],
                data_per_streamline={
                    'seeds': self.starting_points
                })

        # Reshape peaks for displaying
        X, Y, Z, M = self.peaks.get_fdata().shape
        peaks = np.reshape(self.peaks.get_fdata(), (X, Y, Z, 5, M//5))

        # Setup scene and actors
        scene = window.Scene()

        stream_actor = actor.streamtube(tractogram.streamlines)
        peak_actor = actor.peak_slicer(peaks,
                                       np.ones((X, Y, Z, M)),
                                       colors=(0.2, 0.2, 1.),
                                       opacity=0.5)
        dot_actor = actor.dots(tractogram.data_per_streamline['seeds'],
                               color=(1, 1, 1),
                               opacity=1,
                               dot_size=2.5)
        scene.add(stream_actor)
        scene.add(peak_actor)
        scene.add(dot_actor)
        scene.reset_camera_tight(0.95)

        # Save or display scene
        if filename is not None:
            window.snapshot(
                scene,
                fname=filename,
                offscreen=True,
                size=(800, 800))
        else:
            showm = window.ShowManager(scene, reset_camera=True)
            showm.initialize()
            showm.start()









def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value
    return ((flags.astype(np.uint8) & ref_flag) >>
            np.log2(ref_flag).astype(np.uint8)).astype(bool)


class TrackingEnvironment(BaseEnv):
    """ Tracking environment.
    TODO: Clean up "_private functions" and public functions. Some could
    go into BaseEnv.
    """

    def _is_stopping(
        self,
        streamlines: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked

        Returns
        -------
        stopping: `numpy.ndarray`
            Mask of stopping streamlines.
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline.
        """
        # print("_inside _is_stopping fn: streamline sahpe: ", streamlines.shape)#ASHUTOSH
        stopping, flags = \
            self._filter_stopping_streamlines(
                streamlines, self.stopping_criteria)
        return stopping, flags

    def _keep(
        self,
        idx: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """ Keep only states that correspond to continuing streamlines.

        Parameters
        ----------
        idx : `np.ndarray`
            Indices of the streamlines/states to keep
        state: np.ndarray
            Batch of states.

        Returns:
        --------
        state: np.ndarray
            Continuing states.
        """
        state = state[idx]

        return state

    def nreset(self, n_seeds: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines. Will
        chose N random seeds among all seeds.

        Parameters
        ----------
        n_seeds: int
            How many seeds to sample

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Heuristic to avoid duplicating seeds if fewer seeds than actors.
        replace = n_seeds > len(self.seeds)
        seeds = np.random.choice(
            np.arange(len(self.seeds)), size=n_seeds, replace=replace)
        self.initial_points = self.seeds[seeds]

        self.streamlines = np.zeros(
            (n_seeds, self.max_nb_steps + 1, 3), dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points

        self.flags = np.zeros(n_seeds, dtype=int)

        self.lengths = np.ones(n_seeds, dtype=np.int32)

        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(n_seeds, False)
        self.continue_idx = np.arange(n_seeds)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def reset(self, start: int, end: int) -> np.ndarray:
        """ Initialize tracking seeds and streamlines. Will select
        a given batch of seeds.

        Parameters
        ----------
        start: int
            Starting index of seed to add to batch
        end: int
            Ending index of seeds to add to batch

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """
        # Initialize seeds as streamlines
        self.initial_points = self.seeds[start:end]
        # print(self.initial_points.shape)
        N = self.initial_points.shape[0]

        self.streamlines = np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points
        self.flags = np.zeros(N, dtype=int)

        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions, rescale actions to step size and grow streamlines
        for one step forward. Calculate rewards and stop streamlines.

        Parameters
        ----------
        directions: np.ndarray
            Actions applied to the state

        Returns
        -------
        state: np.ndarray
            New state
        reward: list
            Reward for the last step of the streamline
        done: bool
            Whether the episode is done
        info: dict
        """

        # Scale directions to step size
        directions = normalize_vectors(directions) * self.step_size

        # Grow streamlines one step forward
        self.streamlines[self.continue_idx, self.length, :] = \
            self.streamlines[self.continue_idx, self.length-1, :] + directions
        self.length += 1

        # Get stopping and keeping indexes
        stopping, new_flags = \
            self._is_stopping(
                self.streamlines[self.continue_idx, :self.length])
        # print('stopping : ', stopping, '\n','new : ', new_flags)#ASHUTOSH
        # print(type(stopping), type(new_flags))

        # print("is too curvy",is_too_curvy(self.streamlines[self.continue_idx, :self.length] , 80))
        # print("streamlines",self.streamlines[self.continue_idx, :self.length])
        # stopping[0] = False #ASHUTOSh

        self.new_continue_idx, self.stopping_idx = \
            (self.continue_idx[~stopping],
             self.continue_idx[stopping])

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_stopping_idx = np.arange(
            len(self.continue_idx))[~mask_continue]

        self.flags[
            self.stopping_idx] = new_flags[diff_stopping_idx]

        self.dones[self.stopping_idx] = 1

        reward = np.zeros(self.streamlines.shape[0])
        # Compute reward if wanted. At valid time, no need
        # to compute it and slow down the tracking process
        if self.compute_reward:
            reward = self.reward_function(
                self.streamlines[self.continue_idx, :self.length],
                self.dones[self.continue_idx])

        return (
            self._format_state(
                self.streamlines[self.continue_idx, :self.length]),
            reward, self.dones[self.continue_idx],
            {'continue_idx': self.continue_idx})

    def harvest(
        self,
        states: np.ndarray,
    ) -> Tuple[StatefulTractogram, np.ndarray]:
        """Internally keep only the streamlines and corresponding env. states
        that haven't stopped yet, and return the states that continue.

        Parameters
        ----------
        states: torch.Tensor
            States before "pruning" or "harvesting".

        Returns
        -------
        states: np.ndarray of size [n_streamlines, input_size]
            States corresponding to continuing streamlines.
        continue_idx: np.ndarray
            Indexes of trajectories that did not stop.
        """

        # Register the length of the streamlines that have stopped.
        self.lengths[self.stopping_idx] = self.length

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_continue_idx = np.arange(
            len(self.continue_idx))[mask_continue]
        self.continue_idx = self.new_continue_idx

        # Keep only streamlines that should continue
        states = self._keep(
            diff_continue_idx,
            states)

        return states, diff_continue_idx

    def get_streamlines(self) -> StatefulTractogram:
        """ Obtain tracked streamlines fromm the environment.
        The last point will be removed if it raised a curvature stopping
        criteria (i.e. the angle was too high). Otherwise, other last points
        are kept.

        TODO: remove them also ?

        Returns
        -------
        tractogram: Tractogram
            Tracked streamlines.

        """

        tractogram = Tractogram()
        # Harvest stopped streamlines and associated data
        # stopped_seeds = self.first_points[self.stopping_idx]
        # Exclude last point as it triggered a stopping criteria.
        stopped_streamlines = [self.streamlines[i, :self.lengths[i], :]
                               for i in range(len(self.streamlines))]

        flags = is_flag_set(
            self.flags, StoppingFlags.STOPPING_CURVATURE)
        stopped_streamlines = [
            s[:-1] if f else s for f, s in zip(flags, stopped_streamlines)]

        stopped_seeds = self.initial_points

        # Harvested tractogram
        tractogram = Tractogram(
            streamlines=stopped_streamlines,
            data_per_streamline={"seeds": stopped_seeds,
                                 },
            affine_to_rasmm=self.affine)

        return tractogram
    
    #by ashutosh:
    def reset_from_state_coord(self, initial_coord: np.ndarray) -> np.ndarray:
        """ Reset the environment from a given seed point.

        Parameters
        ----------
        initial_coord: np.ndarray
            Initial coordinate for RL model.

        Returns
        -------
        state: np.ndarray
            Updated state for the environment.
        """
        # Extract the initial points from the provided state
        self.initial_points = initial_coord[:, 0, :]

        N = self.initial_points.shape[0]

        # Initialize streamlines based on the provided state
        self.streamlines = np.zeros(
            (N, self.max_nb_steps + 1, 3),
            dtype=np.float32)
        self.streamlines[:, 0, :] = self.initial_points

        # Update other internal variables accordingly
        self.flags = np.zeros(N, dtype=int)
        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Initialize rewards and done flags
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Setup input signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])
    






class RetrackingEnvironment(TrackingEnvironment):
    """ Pre-initialized environment
    Tracking will start from the end of streamlines for two reasons:
        - For computational purposes, it's easier if all streamlines have
          the same length and are harvested as they end
        - Tracking back the streamline and computing the alignment allows some
          sort of "self-supervised" learning for tracking backwards
    """
    def __init__(self, env: TrackingEnvironment, env_dto: dict):

        # Volumes and masks
        self.reference = env.reference
        self.affine = env.affine
        self.affine_rasmm2vox = env.affine_rasmm2vox

        self.data_volume = env.data_volume
        self.tracking_mask = env.tracking_mask
        self.target_mask = env.target_mask
        self.include_mask = env.include_mask
        self.exclude_mask = env.exclude_mask
        self.peaks = env.peaks

        self.normalize_obs = False  # env_dto['normalize']
        self.obs_rms = None

        self._state_size = None  # to be calculated later

        # Tracking parameters
        self.n_signal = env_dto['n_signal']
        self.n_dirs = env_dto['n_dirs']
        self.theta = theta = env_dto['theta']
        self.cmc = env_dto['cmc']
        self.asymmetric = env_dto['asymmetric']

        step_size_mm = env_dto['step_size']
        min_length_mm = env_dto['min_length']
        max_length_mm = env_dto['max_length']
        add_neighborhood_mm = env_dto['add_neighborhood']

        # Reward parameters
        self.alignment_weighting = env_dto['alignment_weighting']
        self.straightness_weighting = env_dto['straightness_weighting']
        self.length_weighting = env_dto['length_weighting']
        self.target_bonus_factor = env_dto['target_bonus_factor']
        self.exclude_penalty_factor = env_dto['exclude_penalty_factor']
        self.angle_penalty_factor = env_dto['angle_penalty_factor']
        self.compute_reward = env_dto['compute_reward']
        self.scoring_data = env_dto['scoring_data']

        self.rng = env_dto['rng']
        self.device = env_dto['device']

        # Stopping criteria is a dictionary that maps `StoppingFlags`
        # to functions that indicate whether streamlines should stop or not
        self.stopping_criteria = {}
        mask_data = env.tracking_mask.get_fdata().astype(np.uint8)

        self.step_size = convert_length_mm2vox(
            step_size_mm,
            self.affine)
        self.min_length = min_length_mm
        self.max_length = max_length_mm

        # Compute maximum length
        self.max_nb_steps = int(self.max_length / step_size_mm)
        self.min_nb_steps = int(self.min_length / step_size_mm)

        if self.compute_reward:
            self.reward_function = Reward(
                peaks=self.peaks,
                exclude=self.exclude_mask,
                target=self.target_mask,
                max_nb_steps=self.max_nb_steps,
                theta=self.theta,
                min_nb_steps=self.min_nb_steps,
                asymmetric=self.asymmetric,
                alignment_weighting=self.alignment_weighting,
                straightness_weighting=self.straightness_weighting,
                length_weighting=self.length_weighting,
                target_bonus_factor=self.target_bonus_factor,
                exclude_penalty_factor=self.exclude_penalty_factor,
                angle_penalty_factor=self.angle_penalty_factor,
                scoring_data=self.scoring_data,
                reference=env.reference)

        self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
            functools.partial(is_too_long,
                              max_nb_steps=self.max_nb_steps)

        self.stopping_criteria[
            StoppingFlags.STOPPING_CURVATURE] = \
            functools.partial(is_too_curvy, max_theta=theta)

        if self.cmc:
            cmc_criterion = CmcStoppingCriterion(
                self.include_mask.get_fdata(),
                self.exclude_mask.get_fdata(),
                self.affine,
                self.step_size,
                self.min_nb_steps)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = cmc_criterion
        else:
            binary_criterion = BinaryStoppingCriterion(
                mask_data,
                0.5)
            self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
                binary_criterion

        # self.stopping_criteria[
        #     StoppingFlags.STOPPING_LOOP] = \
        #     functools.partial(is_looping,
        #                       loop_threshold=300)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood_mm:
            self.add_neighborhood_vox = convert_length_mm2vox(
                add_neighborhood_mm,
                self.affine)
            self.neighborhood_directions = torch.tensor(
                get_neighborhood_directions(
                    radius=self.add_neighborhood_vox),
                dtype=torch.float16).to(self.device)

    @classmethod
    def from_env(
        cls,
        env_dto: dict,
        env: TrackingEnvironment,
    ):
        """ Initialize the environment from a `forward` environment.
        """
        return cls(env, env_dto)

    def _is_stopping(
        self,
        streamlines: np.ndarray,
        is_still_initializing: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Check which streamlines should stop or not according to the
        predefined stopping criteria. An additional check is performed
        to prevent stopping if the retracking process is not over.

        Parameters
        ----------
        streamlines : `numpy.ndarray` of shape (n_streamlines, n_points, 3)
            Streamlines that will be checked
        is_still_initializing: `numpy.ndarray` of shape (n_streamlines)
            Mask that indicates which streamlines are still being
            retracked.

        Returns
        -------
        stopping: `numpy.ndarray`
            Mask of stopping streamlines.
        stopping_flags : `numpy.ndarray`
            `StoppingFlags` that triggered stopping for each stopping
            streamline.
        """
        stopping, flags = super()._is_stopping(streamlines)

        # Streamlines that haven't finished initializing should keep going
        stopping[is_still_initializing[self.continue_idx]] = False
        flags[is_still_initializing[self.continue_idx]] = 0

        return stopping, flags

    def reset(self, half_streamlines: np.ndarray) -> np.ndarray:
        """ Initialize tracking from half-streamlines.

        Parameters
        ----------
        half_streamlines: np.ndarray
            Half-streamlines to initialize environment

        Returns
        -------
        state: numpy.ndarray
            Initial state for RL model
        """

        # Half-streamlines
        self.initial_points = np.array([s[0] for s in half_streamlines])

        # Number if initialization steps for each streamline
        self.n_init_steps = np.asarray(list(map(len, half_streamlines)))

        N = len(self.n_init_steps)

        # Get the first point of each seed as the start of the new streamlines
        self.streamlines = np.zeros(
            (N, self.max_nb_steps, 3),
            dtype=np.float32)

        for i, (s, l) in enumerate(zip(half_streamlines, self.n_init_steps)):
            self.streamlines[i, :l, :] = s[::-1]

        self.seeding_streamlines = self.streamlines.copy()

        self.lengths = np.ones(N, dtype=np.int32)
        self.length = 1

        # Done flags for tracking backwards
        self.flags = np.zeros(N, dtype=int)
        self.dones = np.full(N, False)
        self.continue_idx = np.arange(N)

        # Signal
        return self._format_state(
            self.streamlines[self.continue_idx, :self.length])

    def step(
        self,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, list, bool, dict]:
        """
        Apply actions and grow streamlines for one step forward
        Calculate rewards and if the tracking is done. While tracking
        has not surpassed half-streamlines, replace the tracking step
        taken with the actual streamline segment.

        Parameters
        ----------
        directions: np.ndarray
            Actions applied to the state

        Returns
        -------
        state: np.ndarray
            New state
        reward: list
            Reward for the last step of the streamline
        done: bool
            Whether the episode is done
        info: dict
        """

        # Scale directions to step size
        directions = normalize_vectors(directions) * self.step_size

        # Grow streamlines one step forward
        self.streamlines[self.continue_idx, self.length,
                         :] = self.streamlines[
                             self.continue_idx, self.length-1, :] + directions
        self.length += 1

        # Check which streamline are still being retracked
        is_still_initializing = self.n_init_steps > self.length + 1

        # Get stopping and keeping indexes
        # self._is_stopping is overridden to take into account retracking
        stopping, new_flags = self._is_stopping(
            self.streamlines[self.continue_idx, :self.length],
            is_still_initializing)

        self.new_continue_idx, self.stopping_idx = (
            self.continue_idx[~stopping],
            self.continue_idx[stopping])

        mask_continue = np.in1d(
            self.continue_idx, self.new_continue_idx, assume_unique=True)
        diff_stopping_idx = np.arange(
            len(self.continue_idx))[~mask_continue]

        # Set "done" flags for RL
        self.dones[self.stopping_idx] = 1

        # Store stopping flags
        self.flags[
            self.stopping_idx] = new_flags[diff_stopping_idx]

        # Compute reward
        reward = np.zeros(self.streamlines.shape[0])
        if self.compute_reward:
            # Reward streamline step
            reward = self.reward_function(
                self.streamlines[self.continue_idx, :self.length, :],
                self.dones[self.continue_idx])

        # If a streamline is still being retracked
        if np.any(is_still_initializing):
            # Replace the last point of the predicted streamlines with
            # the seeding streamlines at the same position

            self.streamlines[is_still_initializing, self.length - 1] = \
                self.seeding_streamlines[is_still_initializing,
                                         self.length - 1]

        # Return relevant infos
        return (
            self._format_state(
                self.streamlines[self.continue_idx, :self.length]),
            reward, self.dones[self.continue_idx],
            {'continue_idx': self.continue_idx})


