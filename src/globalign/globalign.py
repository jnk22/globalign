###
### Multimodal registration with exhaustive search mutual information
### Author: Johan \"{O}fverstedt
###

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from numpy import float64, int64

import transformations

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator
    from numpy.typing import NDArray

    from transformations.transformations import CompositeTransform

DEFAULT_EPS: Final = 1e-7


# Creates a list of random angles
def grid_angles(center: int, radius: float, n: int = 32) -> list[float]:
    offsets = np.linspace(-radius, radius, num=n, endpoint=radius < 180)

    return (center + offsets).tolist()


# Supply a Random number generator (e.g. 'rng = np.random.default_rng(12345)') for reproducible results
# Default: 'rng=None' -> np.random.default_rng()
def random_angles(
    centers: list[float] | float,
    center_prob: list[NDArray] | None,
    radius: float,
    n: int = 32,
    rng: Generator | None = None,
) -> list[float64 | float]:
    if not isinstance(centers, list):
        centers = [centers]

    rng = rng or np.random.default_rng()
    p = center_prob / np.sum(center_prob) if center_prob is not None else None

    sampled_centers = rng.choice(np.asarray(centers), size=n, p=p)
    noise = rng.uniform(-radius, radius, size=n)

    return (sampled_centers + noise).tolist()


def compute_entropy(
    C: torch.Tensor, N: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    p = C / N
    return p * torch.log2(torch.clamp(p, min=eps, max=None))


def float_compare(A: torch.Tensor, c: int) -> torch.Tensor:
    return torch.clamp(1 - torch.abs(A - c), 0.0)


def fft_of_levelsets(
    A: torch.Tensor, Q: int, packing: int64, setup_fn: Callable
) -> list[tuple[torch.Tensor, int, int64]]:
    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = np.minimum(a_start + packing, Q)
        levelsets = [float_compare(A, a) for a in range(a_start, a_end)]
        ffts = setup_fn(torch.cat(levelsets, 0))
        fft_list.append((ffts, a_start, a_end))

    return fft_list


def fft(A: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft2(A)


def ifft(Afft: torch.Tensor) -> torch.Tensor:
    return torch.fft.irfft2(Afft)


def fftconv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A * B


def corr_target_setup(A: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft2(A)


def corr_template_setup(B: torch.Tensor) -> torch.Tensor:
    return torch.conj(torch.fft.rfft2(B))


def corr_apply(
    A: torch.Tensor,
    B: torch.Tensor,
    sz: torch.Tensor | tuple[int, ...],
    do_rounding: bool = True,
) -> torch.Tensor:
    C = torch.fft.irfft2(A * B)
    C = C[: sz[0], : sz[1], : sz[2], : sz[3]]

    return torch.round(C) if do_rounding else C


def tf_rotate(
    I: torch.Tensor,
    angle: float | float64,
    fill_value: int,
    center: NDArray | None = None,
) -> torch.Tensor:
    # Half a pixel offset, since TF.rotate origin is in upper left corner.
    center_fixed = [round(x + 0.5) for x in center] if center is not None else center

    return TF.rotate(I, -angle, center=center_fixed, fill=[fill_value])


def create_float_tensor(
    shape: torch.Tensor | tuple[int, ...], on_gpu: bool, fill_value: float | None = None
) -> torch.Tensor:
    device = "cuda" if on_gpu else "cpu"

    # Explicitly convert each tensor element as integer to solve pyright warnings.
    out_shape = tuple(int(dim) for dim in shape)

    return torch.full(
        out_shape, fill_value=fill_value or 0, dtype=torch.float32, device=device
    )


def to_tensor(A: torch.Tensor | NDArray, on_gpu: bool = True) -> torch.Tensor:
    if not isinstance(A, torch.Tensor):
        A = torch.tensor(A, dtype=torch.float32)

    A = A.to(device="cuda", non_blocking=True) if on_gpu else A

    if A.ndim == 2:
        A = A.unsqueeze(0).unsqueeze(0)
    elif A.ndim == 3:
        A = A.unsqueeze(0)

    return A


###
### align_rigid
###
### Performs rigid alignment of multimodal images using exhaustive search mutual information (MI),
### locating the global maximum of the MI measure w.r.t. all possible whole-pixel translations as well
### as a set of enumerated rotations. Runs on the GPU, using PyTorch.
###
### Parameters:
### A: (reference 2d image).
### B: (floating 2d image).
### M_A: (reference 2d mask image).
### M_B: (floating 2d mask image).
### Q_A: (number of quantization levels in image A).
### Q_B: (number of quantization levels in image B).
### angles: List of angles for the rigid alignment.
### overlap: The required overlap fraction (of the maximum overlap possible, given masks).
### enable_partial_overlap: If False then no padding will be done, and only fully overlapping
###    configurations will be evaluated. If True, then padding will be done to include
###    configurations where only part of image B is overlapping image A.
### normalize_mi: Flag to choose between normalized mutual information (NMI) or
###    standard unnormalized mutual information.
### on_gpu: Flag controlling if the alignment is done on the GPU.
### save_maps: Flag for exporting the stack of CMIF maps over the angles, e.g. for debugging or visualization.
### Returns: np.array with 6 values (mutual_information, angle, y, x, y of center of rotation (origin at center of top left pixel), x of center of rotation), maps/None.
###
###  Note: Prior to v1.0.2, the returned center of rotation used Torchvisions convention 'Origin is the upper left corner' which
###   is incompatible with the followed use of 'scipy.ndimage.interpolation.map_coordinates' that assumes integer coordinate-centered pixels.
###
def align_rigid(
    A: NDArray,
    B: NDArray,
    M_A: NDArray | torch.Tensor | None,
    M_B: NDArray | torch.Tensor | None,
    Q_A: int,
    Q_B: int,
    angles: list[float64 | float],
    overlap: float = 0.5,
    enable_partial_overlap: bool = True,
    normalize_mi: bool = False,
    on_gpu: bool = True,
    save_maps: bool = False,
) -> tuple[list, list | None]:
    A_tensor = to_tensor(A, on_gpu=on_gpu)
    B_tensor = to_tensor(B, on_gpu=on_gpu)

    if A_tensor.shape[-1] < 1024:
        packing = np.minimum(Q_B, 64)
    elif A_tensor.shape[-1] <= 2048:
        packing = np.minimum(Q_B, 8)
    elif A_tensor.shape[-1] <= 4096:
        packing = np.minimum(Q_B, 4)
    else:
        packing = np.minimum(Q_B, 1)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = to_tensor(M_A, on_gpu)
        A_tensor = torch.round(M_A * A_tensor + (1 - M_A) * (Q_A + 1))
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = to_tensor(M_B, on_gpu)

    # Pad for overlap
    if enable_partial_overlap:
        pad_y, pad_x = np.round(np.array(B.shape[-2:]) * (1 - overlap)).astype(int)
        pad = (pad_x, pad_x, pad_y, pad_y)

        A_tensor = F.pad(A_tensor, pad, mode="constant", value=Q_A + 1)
        M_A = F.pad(M_A, pad, mode="constant", value=0)
    else:
        pad_y, pad_x = (0, 0)

    ext_ashape = np.array(A_tensor.shape, dtype=int)
    ext_bshape = np.array(B_tensor.shape, dtype=int)
    ext_valid_shape = tuple(ext_ashape - ext_bshape + 1)
    batched_valid_shape = tuple(ext_valid_shape + np.array([packing - 1, 0, 0, 0]))
    y, x = ext_ashape[-2:] - ext_bshape[-2:]

    # use default center of rotation (which is the center point)
    center = transformations.image_center_point(B)

    M_A_FFT = corr_target_setup(M_A)
    A_ffts = [corr_target_setup(float_compare(A_tensor, a)) for a in range(Q_A)]

    if normalize_mi:
        H_MARG = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = create_float_tensor(ext_valid_shape, on_gpu, 0.0)

    results = []
    maps = [] if save_maps else None

    for mi in angles:
        # preprocess B for angle
        m_b_rotated = tf_rotate(M_B, mi, 0, center=center)
        b_rotated = tf_rotate(B_tensor, mi, Q_B, center=center)
        b_rotated = torch.round(m_b_rotated * b_rotated + (1 - m_b_rotated) * (Q_B + 1))
        out_shape = (0, x, 0, y, 0, 0, 0, 0)
        b_rotated = F.pad(b_rotated, out_shape, mode="constant", value=Q_B + 1)
        m_b_rotated = F.pad(m_b_rotated, out_shape, mode="constant", value=0)

        M_B_FFT = corr_template_setup(m_b_rotated)

        corr_0 = corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape)
        N = torch.clamp(corr_0, min=DEFAULT_EPS, max=None)

        b_ffts = fft_of_levelsets(b_rotated, Q_B, packing, corr_template_setup)

        for i, b_fft in enumerate(b_ffts):
            corr_1 = corr_apply(M_A_FFT, b_fft[0], batched_valid_shape)
            E_M = torch.sum(compute_entropy(corr_1, N, DEFAULT_EPS), dim=0)

            if normalize_mi:
                H_MARG = torch.sub(H_MARG, E_M)
            else:
                MI = torch.sub(MI, E_M)

            for a in range(Q_A):
                A_fft_cuda = A_ffts[a]

                if i == 0:
                    corr_2 = corr_apply(A_fft_cuda, M_B_FFT, ext_valid_shape)
                    E_M = compute_entropy(corr_2, N, DEFAULT_EPS)
                    if normalize_mi:
                        H_MARG = torch.sub(H_MARG, E_M)
                    else:
                        MI = torch.sub(MI, E_M)

                corr_3 = corr_apply(A_fft_cuda, b_fft[0], batched_valid_shape)
                E_J = torch.sum(compute_entropy(corr_3, N, DEFAULT_EPS), dim=0)
                if normalize_mi:
                    H_AB = torch.sub(H_AB, E_J)
                else:
                    MI = torch.add(MI, E_J)

        if normalize_mi:
            MI = torch.clamp((H_MARG / (H_AB + DEFAULT_EPS) - 1), 0.0, 1.0)

        if maps is not None:
            maps.append(MI.cpu().numpy())

        max_n, _ = torch.max(torch.reshape(N, (-1,)), 0)
        N_filt = torch.lt(N, overlap * max_n)
        MI[N_filt] = 0.0

        MI_vec = torch.reshape(MI, (-1,))
        results.append((mi, *torch.max(MI_vec, -1)))

        if normalize_mi:
            H_MARG.fill_(0)
            H_AB.fill_(0)
        else:
            MI.fill_(0)

    print("------------------------------")
    print(" [MI]   [angle]  [dx] [dy] ")
    cpu_results = []
    for item in results:
        angle = item[0]
        mi = item[1].cpu().numpy()
        index = item[2].cpu().numpy()
        sz_x = ext_valid_shape[3]
        ty = -(index // sz_x - pad_y)
        tx = -(index % sz_x - pad_x)
        cpu_results.append((mi, angle, ty, tx, center[1], center[0]))

    cpu_results = sorted(cpu_results, key=(lambda res: res[0]), reverse=True)
    for mi, angle, dx, dy, *_ in cpu_results:
        print(f"{mi:.4f} {angle:8.3f} {dx:4d} {dy:4d}")
    print("------------------------------")

    return cpu_results, maps


###
### align_rigid_and_refine
###
### Performs rigid alignment of multimodal images using exhaustive search mutual information (MI),
### locating the global maximum of the MI measure w.r.t. all possible whole-pixel translations as well
### as a set of enumerated rotations. Runs on the GPU, using PyTorch.
###
### Parameters:
### A: (reference 2d image).
### B: (floating 2d image).
### M_A: (reference 2d mask image).
### M_B: (floating 2d mask image).
### Q_A: (number of quantization levels in image A).
### Q_B: (number of quantization levels in image B).
### angles_n: Number of angles to consider in the grid search.
### max_angle: The largest angle to include in the grid search. (180 => global search)
### refinement_param: dictionary with settings for the refinement steps e.g. {'n': 32, 'max_angle': 3.0}
### overlap: The required overlap fraction (of the maximum overlap possible, given masks).
### enable_partial_overlap: If False then no padding will be done, and only fully overlapping
###    configurations will be evaluated. If True, then padding will be done to include
###    configurations where only part of image B is overlapping image A.
### normalize_mi: Flag to choose between normalized mutual information (NMI) or
###    standard unnormalized mutual information.
### on_gpu: Flag controlling if the alignment is done on the GPU.
### save_maps: Flag for exporting the stack of CMIF maps over the angles, e.g. for debugging or visualization.
### rng: Optional random number generator (e.g. 'rng = np.random.default_rng(12345)') for reproducible results; default: None -> np.random.default_rng()
### Returns: np.array with 6 values (mutual_information, angle, y, x, y of center of rotation, x of center of rotation), maps/None.
###
def align_rigid_and_refine(
    A: NDArray,
    B: NDArray,
    M_A: NDArray,
    M_B: NDArray,
    Q_A: int,
    Q_B: int,
    angles_n: int,
    max_angle: float,
    refinement_param: dict[str, Any] | None = None,
    overlap: float = 0.5,
    enable_partial_overlap: bool = True,
    normalize_mi: bool = False,
    on_gpu: bool = True,
    save_maps: bool = False,
    rng: Generator | None = None,
) -> tuple[NDArray, tuple[list | None, ...]]:
    if refinement_param is None:
        refinement_param = {"n": 32}

    args = A, B, M_A, M_B, Q_A, Q_B
    kwargs = {
        "overlap": overlap,
        "enable_partial_overlap": enable_partial_overlap,
        "normalize_mi": normalize_mi,
        "on_gpu": on_gpu,
        "save_maps": save_maps,
    }

    start_angles = grid_angles(0, max_angle, n=angles_n)
    start_results, start_maps = align_rigid(*args, angles=start_angles, **kwargs)
    best_result = start_results[0]

    # Extract rotations and probabilities for refinement.
    centers, center_probs = best_result[1], [best_result[0]]

    n = refinement_param.get("n", 0)
    if n <= 0:
        return np.array(best_result), (start_maps,)

    angle_limit = refinement_param.get("max_angle", 3.0)
    refine_angles = random_angles(centers, center_probs, angle_limit, n=n, rng=rng)
    refine_results, refine_maps = align_rigid(*args, angles=refine_angles, **kwargs)

    max_result = max([best_result, refine_results[0]], key=(lambda res: res[0]))

    return np.array(max_result), (start_maps, refine_maps)


###
### warp_image_rigid
###
### Applies the transformation obtained by the functions align_rigid/align_rigid_and_refine
### to warp a floating image into the space of the ref_image (using backward mapping).
### param: The parameters (first value of the returned tuple from align_rigid/align_rigid_and_refine)
### mode (interpolation): nearest/linear/spline
### bg_value: The value to insert where there is no information in the flo_image
### inv: Invert the transformation, used e.g. when warping the original reference image into
###      the space of the original floating image.
def warp_image_rigid(
    ref_image: NDArray,
    flo_image: NDArray,
    param: NDArray,
    mode: str = "nearest",
    bg_value: list[float] | float = 0.0,
    inv: bool = False,
) -> NDArray:
    tform = __create_transformation(param, inv=inv)

    out_shape = ref_image.shape[:2] + flo_image.shape[2:]
    flo_image_out = np.zeros(out_shape, dtype=flo_image.dtype)
    if flo_image.ndim == 3:
        for i in range(flo_image.shape[2]):
            bg_val_i = np.array(bg_value)
            if bg_val_i.shape[0] == flo_image.shape[2]:
                bg_val_i = bg_val_i[i]

            tform.warp(
                flo_image[:, :, i],
                flo_image_out[:, :, i],
                in_spacing=np.ones(2),
                out_spacing=np.ones(2),
                mode=mode,
                bg_value=bg_val_i,
            )
    else:
        tform.warp(
            flo_image,
            flo_image_out,
            in_spacing=np.ones(2),
            out_spacing=np.ones(2),
            mode=mode,
            bg_value=bg_value,
        )

    return flo_image_out


###
### warp_points_rigid
###
### Applies the transformation obtained by the functions align_rigid/align_rigid_and_refine
### to transform a set of points in the reference image space into the floating image space.
### param: The parameters (first value of the returned tuple from align_rigid/align_rigid_and_refine)
### inv: Invert the transformation, used e.g. when transforming points from the original floating image
###      space into the original reference image space.
def warp_points_rigid(points, param, inv: bool = False):
    return __create_transformation(param, inv=inv).transform(points)


def __create_transformation(param, *, inv: bool = False) -> CompositeTransform:
    translation = transformations.TranslationTransform(2)
    translation.set_param(0, param[2])
    translation.set_param(1, param[3])

    rotation = transformations.Rotate2DTransform()
    rotation.set_param(0, np.pi * param[1] / 180.0)

    center = np.array(param[4:])

    tform = transformations.CompositeTransform(2, [translation, rotation])
    tform = transformations.make_centered_transform(tform, center, center)

    return tform.invert() if inv else tform
