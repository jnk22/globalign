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

import transformations

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from numpy.random import Generator as RandomGenerator
    from numpy.typing import NDArray

    from transformations.transformations import CompositeTransform

EPS: Final = 1e-7
SEPARATOR: Final = "-" * 30
HEADER: Final = " [MI]   [angle]  [dx] [dy] "


# Creates a list of random angles
def grid_angles(center: float, radius: float, n: int = 32) -> list[float]:
    if radius < 0:
        msg = "radius must be >= 0"
        raise ValueError(msg)
    if n < 1:
        return []

    offsets = np.linspace(-radius, radius, num=n, endpoint=radius < 180)

    return (center + offsets).tolist()


# Supply a Random number generator (e.g. 'rng = np.random.default_rng(12345)') for reproducible results
# Default: 'rng=None' -> np.random.default_rng()
def random_angles(
    centers: list[float] | float,
    center_prob: list[NDArray] | None,
    radius: float,
    n: int = 32,
    rng: RandomGenerator | None = None,
) -> list[float]:
    if radius < 0:
        msg = "radius must be >= 0"
        raise ValueError(msg)
    if n < 1:
        return []

    centers_vector = np.atleast_1d(centers)

    if center_prob is not None:
        if len(center_prob) != len(centers_vector):
            msg = "centers and center_prob must have same size"
            raise ValueError(msg)

        p = center_prob / np.sum(center_prob)
    else:
        p = None

    rng = rng or np.random.default_rng()
    sampled_centers = rng.choice(centers_vector, size=n, p=p)
    noise = rng.uniform(-radius, radius, size=n)

    return (sampled_centers + noise).tolist()


def compute_entropy(
    C: torch.Tensor, N: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    p = C / N
    return p * torch.log2(torch.clamp(p, min=eps))


def float_compare(A: torch.Tensor, c: int) -> torch.Tensor:
    return F.relu(1 - torch.abs(A - c))


def fft_of_levelsets(
    A: torch.Tensor, Q: int, packing: int, setup_fn: Callable
) -> list[tuple[torch.Tensor, int, int]]:
    shape = (1,) * (A.ndim - 1)
    arange = torch.arange(0, Q, device=A.device, dtype=A.dtype).view(Q, *shape)

    levelsets_all = F.relu(1 - torch.abs(A - arange))

    fft_list = []
    for a_start in range(0, Q, packing):
        a_end = min(a_start + packing, Q)

        fft_list.append((setup_fn(levelsets_all[a_start:a_end]), a_start, a_end))

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
    C = torch.fft.irfft2(A * B)[: sz[0], : sz[1], : sz[2]]
    return torch.round(C) if do_rounding else C


def tf_rotate(
    I: torch.Tensor, angle: float, fill_value: int, center: NDArray | None = None
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


def to_tensor(
    A: torch.Tensor | NDArray, on_gpu: bool = True, *, target_dim: int = 4
) -> torch.Tensor:
    device = "cuda" if on_gpu else "cpu"

    A = (
        A.to(device=device, non_blocking=True)
        if isinstance(A, torch.Tensor)
        else torch.tensor(A, dtype=torch.float32, device=device)
    )

    while A.ndim < target_dim:
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
### packing: Maximum parallel FFT operations.
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
    angles: list[float],
    overlap: float = 0.5,
    enable_partial_overlap: bool = True,
    normalize_mi: bool = False,
    packing: int | None = None,
    on_gpu: bool = True,
    save_maps: bool = False,
) -> tuple[list, list | None]:
    a_tensor = to_tensor(A, on_gpu=on_gpu, target_dim=3)
    b_tensor = to_tensor(B, on_gpu=on_gpu, target_dim=3)

    if packing is None:
        # Use default packing to reduce memory usage.
        if a_tensor.shape[-1] < 1024:
            packing = min(Q_B, 64)
        elif a_tensor.shape[-1] <= 2048:
            packing = min(Q_B, 8)
        elif a_tensor.shape[-1] <= 4096:
            packing = min(Q_B, 4)
        else:
            packing = min(Q_B, 1)
    else:
        # packing must be >= 1, but more than Q_B is not necessary.
        packing = max(min(Q_B, packing), 0)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = torch.ones_like(a_tensor)
    else:
        M_A = to_tensor(M_A, on_gpu, target_dim=3)
        a_tensor = torch.round(M_A * a_tensor + (1 - M_A) * (Q_A + 1))

    M_B = (
        torch.ones_like(b_tensor)
        if M_B is None
        else to_tensor(M_B, on_gpu, target_dim=3)
    )

    # Pad for overlap
    if enable_partial_overlap:
        pad_y, pad_x = np.round(np.array(B.shape[-2:]) * (1 - overlap)).astype(int)
        pad = (pad_x, pad_x, pad_y, pad_y)

        a_tensor = F.pad(a_tensor, pad, mode="constant", value=Q_A + 1)
        M_A = F.pad(M_A, pad, mode="constant", value=0)
    else:
        pad_y, pad_x = (0, 0)

    ext_ashape = np.array(a_tensor.shape, dtype=int)
    ext_bshape = np.array(b_tensor.shape, dtype=int)
    ext_shape = tuple(ext_ashape - ext_bshape + 1)
    batch_shape = tuple(ext_shape + np.array([packing - 1, 0, 0]))
    y, x = ext_ashape[-2:] - ext_bshape[-2:]
    out_shape = (0, x, 0, y, 0, 0)
    device = a_tensor.device

    # use default center of rotation (which is the center point)
    center = transformations.image_center_point(B)

    ma_fft = torch.fft.rfft2(M_A)
    arange = torch.arange(0, Q_A, device=device, dtype=a_tensor.dtype)
    shape = (1,) * a_tensor.ndim
    a_ffts = torch.fft.rfft2(F.relu(1 - torch.abs(a_tensor - arange.view(Q_A, *shape))))

    dtype = torch.float32
    mi = torch.zeros(ext_shape, dtype=dtype, device=device)
    h_ab = torch.zeros(ext_shape, dtype=dtype, device=device) if normalize_mi else None

    temp_results = []
    maps: list[NDArray] | None = [] if save_maps else None

    for angle in angles:
        mb_rotated = tf_rotate(M_B, angle, 0, center=center)
        b_rotated = tf_rotate(b_tensor, angle, Q_B, center=center)
        b_rotated = torch.round(mb_rotated * b_rotated + (1 - mb_rotated) * (Q_B + 1))

        mb_rotated = F.pad(mb_rotated, out_shape, mode="constant", value=0)
        b_rotated = F.pad(b_rotated, out_shape, mode="constant", value=Q_B + 1)

        mb_fft = torch.conj(torch.fft.rfft2(mb_rotated))
        n = torch.clamp(corr_apply(ma_fft, mb_fft, ext_shape), min=EPS)

        b_ffts = __fft_of_levelsets(b_rotated, Q_B, packing)

        for i, b_fft in enumerate(b_ffts):
            mi -= torch.sum(__entropy(ma_fft, b_fft, n, shape=batch_shape), dim=0)

            for a_fft in a_ffts:
                if i == 0:
                    mi -= __entropy(a_fft, mb_fft, n, shape=ext_shape)

                mi += torch.sum(__entropy(a_fft, b_fft, n, shape=batch_shape), dim=0)

        if h_ab is not None:
            mi = F.relu(mi / (h_ab + EPS) - 1)

        if maps is not None:
            maps.append(mi.cpu().numpy())

        max_n, _ = torch.max(torch.reshape(n, (-1,)), 0)
        mi[n < overlap * max_n] = 0.0

        mi_vec = torch.reshape(mi, (-1,))
        temp_results.append((angle, *torch.max(mi_vec, -1)))

        mi.zero_()
        if h_ab is not None:
            h_ab.zero_()

    results = []
    for angle, mi, index in temp_results:
        idx = index.cpu().numpy()
        ty = -(idx // ext_shape[2] - pad_y)
        tx = -(idx % ext_shape[2] - pad_x)
        results.append((mi.cpu().numpy(), angle, ty, tx, center[1], center[0]))

    results = sorted(results, key=(lambda res: res[0]), reverse=True)
    lines = (f"{mi:.4f} {ang:8.3f} {dx:4d} {dy:4d}" for mi, ang, dx, dy, *_ in results)
    print("\n".join([SEPARATOR, HEADER, *lines, SEPARATOR]))

    return results, maps


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
    M_A: NDArray | torch.Tensor | None,
    M_B: NDArray | torch.Tensor | None,
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
    rng: RandomGenerator | None = None,
    packing: int | None = None,
) -> tuple[NDArray, tuple[list | None, ...]]:
    if refinement_param is None:
        refinement_param = {"n": 32}

    kwargs: dict[str, Any] = {
        "overlap": overlap,
        "enable_partial_overlap": enable_partial_overlap,
        "normalize_mi": normalize_mi,
        "on_gpu": on_gpu,
        "save_maps": save_maps,
        "packing": packing,
    }

    start_angles = grid_angles(0, max_angle, n=angles_n)
    start_results, start_maps = align_rigid(
        A, B, M_A, M_B, Q_A, Q_B, angles=start_angles, **kwargs
    )
    best_result = start_results[0]

    # Extract rotations and probabilities for refinement.
    centers, center_probs = best_result[1], [best_result[0]]

    n = refinement_param.get("n", 0)
    if n <= 0:
        return np.array(best_result), (start_maps,)

    angle_limit = refinement_param.get("max_angle", 3.0)
    refine_angles = random_angles(centers, center_probs, angle_limit, n=n, rng=rng)
    refine_results, refine_maps = align_rigid(
        A, B, M_A, M_B, Q_A, Q_B, angles=refine_angles, **kwargs
    )

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
def warp_points_rigid(points: NDArray, param: NDArray, inv: bool = False) -> NDArray:
    return __create_transformation(param, inv=inv).transform(points)


def __fft_of_levelsets(
    a: torch.Tensor, q: int, packing: int
) -> Generator[torch.Tensor, None, None]:
    shape = (1,) * (a.ndim - 1)
    arange = torch.arange(0, q, device=a.device, dtype=a.dtype).view(q, *shape)

    levelsets_all = F.relu(1 - torch.abs(a - arange))

    return (
        torch.conj(torch.fft.rfft2(levelsets_all[a_start : min(a_start + packing, q)]))
        for a_start in range(0, q, packing)
    )


def __entropy(
    a: torch.Tensor,
    b: torch.Tensor,
    n: torch.Tensor,
    *,
    shape: tuple[int, ...],
    do_rounding: bool = True,
) -> torch.Tensor:
    c = torch.fft.irfft2(a * b)[: shape[0], : shape[1], : shape[2]]
    p = (torch.round(c) if do_rounding else c) / n

    return p * torch.log2(torch.clamp(p, min=EPS))


def __create_transformation(param: NDArray, *, inv: bool = False) -> CompositeTransform:
    translation = transformations.TranslationTransform(2)
    translation.set_param(0, param[2])
    translation.set_param(1, param[3])

    rotation = transformations.Rotate2DTransform()
    rotation.set_param(0, np.pi * param[1] / 180.0)

    center = np.array(param[4:])

    tform = transformations.CompositeTransform(2, [translation, rotation])
    tform = transformations.make_centered_transform(tform, center, center)

    return tform.invert() if inv else tform
