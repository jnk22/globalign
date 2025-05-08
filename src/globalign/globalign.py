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

__EPS: Final = 1e-7
__SEPARATOR: Final = "-" * 30
__HEADER: Final = " [MI]   [angle]  [dx] [dy] "


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
) -> tuple[NDArray, tuple[list[NDArray] | None, ...]]:
    """Align two 2D images using exhaustive MI-based search with refinement.

    Perform rigid alignment of multimodal images using exhaustive search mutual
    information (MI), locating the global maximum of the MI measure with
    respect to all possible whole-pixel translations and a set of enumerated
    rotations.

    Parameters
    ----------
    A
        The reference image.
    B
        The floating image.
    M_A
        The reference image mask.
    M_B
        The floating image mask.
    Q_A
        The number of quantization levels in image A.
    Q_B
        The number of quantization levels in image B.
    angles_n
        The number of angles to consider in the grid search.
    max_angle
        The largest angle to include in the grid search (180 for a global
        search).
    refinement_param
        A dictionary with settings for the refinement steps, e.g.,
        `{'n': 32, 'max_angle': 3.0}`.
    overlap
        The required overlap fraction (of the maximum overlap possible, given
        the masks).
    enable_partial_overlap
        If False, no padding will be done, and only fully overlapping
        configurations will be evaluated. If True, padding will be done to
        include configurations where only part of image B overlaps with image
        A. Default is False.
    normalize_mi
        Flag to choose between normalized mutual information (NMI) or standard
        unnormalized mutual information. Default is True.
    on_gpu
        If True, the alignment is done on the GPU. Default is True.
    save_maps
        If True, exports the stack of CMIF maps over the angles for debugging
        or visualization. Default is False.
    rng
        An optional random number generator (e.g., `rng =
        np.random.default_rng(12345)`) for reproducible results. Default is
        None (uses `np.random.default_rng()`).

    Returns
    -------
    np.ndarray
        A 1D array with six values: mutual information, angle, y, x, y of
        center of rotation (origin at the center of the top-left pixel), and x
        of center of rotation.
    tuple[list[np.ndarray] | None, ...]]
        Stacks of CMIF maps over the angles for debugging, or None.
    """
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
) -> tuple[list, list[NDArray] | None]:
    """Align two 2D images using exhaustive search based on mutual information (MI).

    This function performs rigid alignment of multimodal images using an
    exhaustive search of mutual information (MI), locating the global maximum
    of the MI measure with respect to all possible whole-pixel translations and
    a set of enumerated rotations.

    Parameters
    ----------
    A
        The reference image.
    B
        The floating image.
    M_A
        The reference image mask.
    M_B
        The floating image mask.
    Q_A
        The number of quantization levels in image A.
    Q_B
        The number of quantization levels in image B.
    angles
        The list of angles for the rigid alignment.
    overlap
        The required overlap fraction (of the maximum overlap possible, given
        the masks).
    enable_partial_overlap
        If False, only fully overlapping configurations are evaluated. If True,
        padding is done to include configurations where only part of image B
        overlaps with image A. Default is False.
    normalize_mi
        If True, use normalized mutual information (NMI), otherwise use
        unnormalized mutual information. Default is True.
    packing
        The maximum number of parallel FFT operations. Default is automatically
        chosen based on image size.
    on_gpu
        If True, the alignment is done on the GPU. Default is True.
    save_maps
        If True, exports the stack of CMIF maps over the angles for debugging
        or visualization. Default is False.

    Returns
    -------
    np.ndarray
        A 1D array with six values: mutual information, angle, y, x, y of
        center of rotation (origin at the center of the top-left pixel), and x
        of center of rotation.
    list[np.ndarray] | None
        Stack of CMIF maps over the angles for debugging, or None.

    Notes
    -----
    Prior to v1.0.2, the returned center of rotation used Torchvision's
    convention ('origin is the upper left corner'), which was incompatible with
    `scipy.ndimage.interpolation.map_coordinates`, which assumes integer
    coordinate-centered pixels.
    """
    device = "cuda" if on_gpu else "cpu"

    a_tensor = __to_tensor(A, device=device)
    b_tensor = __to_tensor(B, device=device)

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
        M_A = torch.ones_like(a_tensor, dtype=torch.float32, device=device)
    else:
        M_A = __to_tensor(M_A, device=device)
        a_tensor = torch.round(M_A * a_tensor + (1 - M_A) * (Q_A + 1))

    if M_B is None:
        M_B = torch.ones_like(b_tensor, dtype=torch.float32, device=device)
    else:
        M_B = __to_tensor(M_B, device=device)

    # Pad for overlap
    if enable_partial_overlap:
        # TODO: Ensure that x/y order is correct.
        pad_y, pad_x = np.round(np.array(B.shape) * (1 - overlap)).astype(int)
        pad_shape = (1, a_tensor.shape[2] + 2 * pad_x, a_tensor.shape[1] + 2 * pad_y)
        overlap_pad_indices = (..., slice(pad_x, -pad_x), slice(pad_y, -pad_y))

        tmp_tensor = torch.full(pad_shape, Q_A + 1, dtype=torch.float32, device=device)
        tmp_tensor[overlap_pad_indices] = a_tensor
        a_tensor = tmp_tensor

        tmp_mask = torch.zeros(pad_shape, dtype=torch.float32, device=device)
        tmp_mask[overlap_pad_indices] = M_A
        M_A = tmp_mask

    else:
        pad_y, pad_x = (0, 0)
        pad_shape = (1, a_tensor.shape[2], a_tensor.shape[1])

    shape_diff = np.array(a_tensor.shape) - np.array(b_tensor.shape)
    ext_shape = tuple(shape_diff + 1)
    ext_indices = tuple(slice(None, ext_shape[i]) for i in range(3))
    pad_indices = (..., slice(None, shape_diff[2]), slice(None, shape_diff[1]))
    batch_shape = (packing, *ext_shape[1:])

    # Use default center of rotation (which is the center point) with
    # half a pixel offset, since TF.rotate origin is in upper left corner.
    center = transformations.image_center_point(B)
    rotation_center = (center + 0.5).tolist()

    ma_fft = torch.fft.rfft2(M_A)
    arange = torch.arange(0, Q_A, dtype=a_tensor.dtype, device=device).reshape(
        Q_A, 1, 1
    )
    a_ffts = torch.fft.rfft2(torch.relu(1 - torch.abs(a_tensor - arange)))

    temp_results = []
    maps: list[NDArray] | None = [] if save_maps else None

    tmp_b_pad = torch.full(pad_shape, Q_B + 1, dtype=torch.float32, device=device)
    tmp_mb_pad = torch.zeros(pad_shape, dtype=torch.float32, device=device)

    for ang in angles:
        mi = torch.zeros(ext_shape, dtype=torch.float32, device=device)
        h_ab = (
            torch.zeros(ext_shape, dtype=torch.float32, device=device)
            if normalize_mi
            else None
        )

        mb_rotated = TF.rotate(M_B, -ang, center=rotation_center, fill=[0])
        b_rotated = TF.rotate(b_tensor, -ang, center=rotation_center, fill=[Q_B])
        b_rotated = torch.round(mb_rotated * b_rotated + (1 - mb_rotated) * (Q_B + 1))

        tmp_b_pad[pad_indices] = b_rotated
        b_rotated = tmp_b_pad

        tmp_mb_pad[pad_indices] = mb_rotated
        mb_rotated = tmp_mb_pad

        mb_fft = torch.conj(torch.fft.rfft2(mb_rotated))

        c = torch.fft.irfft2(ma_fft * mb_fft)[ext_indices]
        n = torch.clamp(torch.round(c), min=__EPS)

        b_ffts = __fft_of_levelsets(b_rotated, Q_B, packing)

        for i, b_fft in enumerate(b_ffts):
            mi -= torch.sum(__entropy(ma_fft, b_fft, n, shape=batch_shape), dim=0)

            for a_fft in a_ffts:
                if i == 0:
                    mi -= __entropy(a_fft, mb_fft, n, shape=ext_shape)

                mi += torch.sum(__entropy(a_fft, b_fft, n, shape=batch_shape), dim=0)

        if h_ab is not None:
            mi = torch.relu(mi / (h_ab + __EPS) - 1)

        if maps is not None:
            maps.append(mi.cpu().numpy())

        # Mask values in `mi` where n < overlap * max_n
        max_n = torch.max(torch.reshape(n, (-1,)))
        mask = torch.less(n, overlap * max_n)
        mi = torch.where(mask, 0.0, mi)

        mi_vec = torch.reshape(mi, (-1,))
        temp_results.append((ang, float(torch.max(mi_vec)), int(torch.argmax(mi_vec))))

    results = [
        (
            mi,
            angle,
            -(index // ext_shape[2] - pad_y),
            -(index % ext_shape[2] - pad_x),
            *center[::-1],
        )
        for angle, mi, index in temp_results
    ]

    results = sorted(results, key=(lambda res: res[0]), reverse=True)
    lines = (f"{mi:.4f} {ang:8.3f} {dx:4d} {dy:4d}" for mi, ang, dx, dy, *_ in results)
    print("\n".join([__SEPARATOR, __HEADER, *lines, __SEPARATOR]))

    return results, maps


def grid_angles(center: float, radius: float, n: int = 32) -> list[float]:
    """Generate a list of angles around a center, within a specified radius.

    Parameters
    ----------
    center
        The central angle.
    radius
        The radius defining the range of angles.
    n
        The number of angles to generate (default is 32).

    Returns
    -------
    list[float]
        A list of angles evenly spaced within the specified range.

    Raises
    ------
    ValueError
        If `radius` is negative.
    """
    if radius < 0:
        msg = "radius must be >= 0"
        raise ValueError(msg)
    if n < 1:
        return []

    offsets = np.linspace(-radius, radius, num=n, endpoint=radius < 180)

    return (center + offsets).tolist()


def random_angles(
    centers: list[float] | float,
    center_prob: list[NDArray] | None,
    radius: float,
    n: int = 32,
    rng: RandomGenerator | None = None,
) -> list[float]:
    """Generate a list of random angles based on given centers, probabilities, and radius.

    Parameters
    ----------
    centers
        The central angles from which to sample.
    center_prob
        The probabilities associated with each center. Must match the length of
        `centers`. If `None`, centers are sampled uniformly.
    radius
        The radius within which the random angles will vary.
    n
        The number of random angles to generate (default is 32).
    rng
        A random number generator to use. If `None`, the default RNG is used.

    Returns
    -------
    list[float]
        A list of `n` random angles.

    Raises
    ------
    ValueError
        If `radius` is negative, or if `center_prob` does not match the size of
        `centers`.
    """
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


def warp_image_rigid(
    ref_image: NDArray,
    flo_image: NDArray,
    param: NDArray,
    mode: str = "nearest",
    bg_value: list[float] | float = 0.0,
    inv: bool = False,
) -> NDArray:
    """Warp a floating image into the space of a reference image.

    Applies the transformation obtained from `align_rigid` or `align_rigid_and_refine`
    to warp the floating image (`flo_image`) into the coordinate space of the reference
    image (`ref_image`) using backward mapping.

    Parameters
    ----------
    ref_image
        The reference image, defining the target coordinate space.
    flo_image
        The floating image to be warped.
    param
        The transformation parameters, typically the first element of the tuple returned
        by `align_rigid` or `align_rigid_and_refine`.
    mode
        Interpolation mode: one of `'nearest'`, `'linear'`, `'cubic'`, or
        `'spline'`. Default is `'nearest'`.
    bg_value
        The background value to use where the floating image does not cover the
        output. Can be a single float or a list of floats (for multi-channel
        images). Default is 0.0.
    inv
        If True, the transformation is inverted. This is useful when warping
        the reference image into the space of the floating image. Default is
        False.

    Returns
    -------
    np.ndarray
        The warped floating image, aligned to the reference image's space.
    """
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


def warp_points_rigid(points: NDArray, param: NDArray, inv: bool = False) -> NDArray:
    """Transform points between image spaces using a rigid transformation.

    Applies the transformation obtained from `align_rigid` or
    `align_rigid_and_refine` to transform a set of points from the reference
    image space into the floating image space (or vice versa, if inverted).

    Parameters
    ----------
    points
        An array of points to be transformed.
    param
        The transformation parameters, typically the first element of the tuple
        returned by `align_rigid` or `align_rigid_and_refine`.
    inv
        If True, inverts the transformation. This is useful when transforming
        points from the floating image space back into the reference image
        space. Default is False.

    Returns
    -------
    np.ndarray
        The transformed points, in the target image space.
    """
    return __create_transformation(param, inv=inv).transform(points)


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
    arange = torch.arange(0, Q, dtype=A.dtype, device=A.device).view(Q, *shape)

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
    center_fixed = (center + 0.5).tolist() if center is not None else center

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


def __fft_of_levelsets(
    a: torch.Tensor, q: int, packing: int
) -> Generator[torch.Tensor, None, None]:
    arange = torch.arange(0, q, dtype=a.dtype, device=a.device).reshape(q, 1, 1)
    levelsets_all = torch.relu(1 - torch.abs(a - arange))

    return (
        torch.conj(torch.fft.rfft2(levelsets_all[a_start : min(a_start + packing, q)]))
        for a_start in range(0, q, packing)
    )


def __to_tensor(arr: torch.Tensor | NDArray, *, device: str) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


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

    return p * torch.log2(torch.clamp(p, min=__EPS))


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
