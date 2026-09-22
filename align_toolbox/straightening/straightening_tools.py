from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import skimage
from numpy.typing import NDArray
from scipy.interpolate import BSpline, splprep
from scipy.sparse import csgraph
from scipy.spatial import distance
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.util import img_as_ubyte

if TYPE_CHECKING:
    from os import PathLike

    from scipy import sparse

# spacing as seen in Ti2 image metadata
DEFAULT_SPACING = np.asarray((2.0, 0.325, 0.325))

RIDGE_DETECTOR: cv2.ximgproc.RidgeDetectionFilter = (
    cv2.ximgproc.RidgeDetectionFilter.create()
)


class Warper:
    """Class for generating, storing, and applying splines fit to midlines of oblong objects for use in image/coordinate warping
    e.g. on worms, worm pharynxes

    individual splines are fit to each plane independently to account for movement during stack acquisition
    """

    def __init__(self, length: float, width: float, splines: list[BSpline | None]):
        """
        Store the warp geometry and per-plane splines.

        Parameters:
            length (float): Length of the straightened object (in pixels along the
                midline axis).
            width (float): Width of the straightened object (in pixels across the
                midline axis).
            splines (list[BSpline or None]): Fitted spline for each z-plane.
                ``None`` entries correspond to empty or unprocessable planes.
        """
        self.length = length
        self.width = width
        self.splines = splines

    @classmethod
    def from_img(cls, img: NDArray, mask: NDArray[np.bool_]) -> Warper:
        """
        Construct a Warper by extracting midlines from a mask and fitting splines.

        Extracts the midline for each plane of ``mask``, ensures consistent
        head/tail orientation across planes, fits splines, and aligns splines
        using image-based registration.

        Parameters:
            img (NDArray): Grayscale image (2D or 3D stack) used for spline
                alignment across planes.
            mask (NDArray[bool]): Binary mask with the same shape as ``img``.

        Returns:
            Warper: A Warper instance ready for image warping.

        Raises:
            ValueError: If ``img`` and ``mask`` shapes do not match, or if the
                mask is invalid (see :func:`validate_mask`).
        """
        if img.shape != mask.shape:
            raise ValueError(
                f"Image and Mask shapes don't match: img.shape = {img.shape}, mask.shape = {mask.shape}"
            )

        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
            mask = mask[np.newaxis, ...]

        validate_mask(mask)

        mask = mask.astype(bool)

        # midlines
        mls = []
        # distance transforms
        dts = []
        for mask_plane in mask:
            if not mask_plane.any():
                mls.append(None)
                dts.append(None)
                continue

            ml, dt = extract_midline(mask_plane, return_dt=True)
            mls.append(ml)
            dts.append(dt)

        mls = _handle_flips(mls)

        splines = []
        spline_lengths = []
        # spline parameter "u" a la "uv-mapping"
        spline_us = []
        for ml in mls:
            # need at least 4 points for spline fitting
            if (ml is None) or (len(ml) < 4):
                splines.append(None)
                spline_lengths.append(None)
                spline_us.append(None)
                continue
            # raw spline only smooths and does not account for inter-pixel distances
            s = len(ml) - np.sqrt(2 * len(ml))
            raw_spline = _fit_spline(ml, parametrisation=None, smoothing_factor=s)
            # refined spline accounts for inter-pixel distances
            refined_spline, spline_length, refined_us = _refine_spline(raw_spline)
            splines.append(refined_spline)
            spline_lengths.append(spline_length)
            spline_us.append(refined_us)

        # dt.max() is a radius, want diameter
        # TODO: is mean sufficient? ideally it's max, but occasionally poor masks REALLY inflate max of dts
        # I sometimes see clipping of elements wider than the mean (especially pharynx bulbs), so I multiply it by 1.2

        worm_width = (np.mean([dt.max() for dt in dts if dt is not None]) * 1.2) * 2
        worm_length = max(length for length in spline_lengths if length is not None)

        # if provided image is 2D, no alignment is necessary
        if len(splines) == 1:
            return cls(worm_length, worm_width, splines)
        # if 3D, proceed to spline alignment

        # handle the missing masks/splines
        not_missing = [item is not None for item in splines]
        good_img = img[not_missing]

        aligned_splines, worm_length = _align_splines(
            good_img,
            worm_width,
            worm_length,
            [spline for spline in splines if spline is not None],
            [us for us in spline_us if us is not None],
        )
        splines = []
        warnings = []
        for i, is_ok in enumerate(not_missing):
            if is_ok:
                splines.append(aligned_splines.pop(0))
            else:
                splines.append(None)
                warnings.append(i)
        if len(warnings) > 0:
            logging.warning(
                f"Masks for planes {warnings} were empty and no splines could be fit. Warped images will be blank"
            )
        return cls(worm_length, worm_width, splines)

    def to_pickle(self, file: PathLike):
        """save object instace to a pickle file"""
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file: PathLike) -> Warper:
        """load a Warper object from a pickle file
        WARNING do not load untrusted pickle files as they can execute arbitrary code
        """
        with open(file, "rb") as f:
            wt = pickle.load(f)
        return wt

    def warp_2D_img(
        self,
        img2D: NDArray,
        spline_i: int,
        scale_factor: float | tuple[float, float] = 1,
        mirror: bool = False,
        interpolation_order: Literal[0, 1, 2, 3, 4, 5] = 1,
        preserve_range: bool = True,
        preserve_dtype: bool = True,
    ) -> NDArray:
        """Warp a single 2D plane using self.splines[i]

        self.splines[i] is the spline fit to mask[i] in the provided mask during object instantiation,
        so usually spline_i should match z of chosen 2D plane from zstack"""
        # scale_factor: how many pixels in straightened image correspond to one pixel in raw
        # mirror: whether to flip the axis perpendicular to spline length.original MATLAB code flipped
        #   practically speaking, if mirror=True, the left-right axis of a worm as seen in image will be flipped
        # interpolation_order:
        # 0: Nearest-neighbor (e.g. use for masks/labels)
        # 1: Bi-linear (default)
        # 2: Bi-quadratic
        # 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic
        # preserve_range: internally, for interpolation, images are always converted to float. if true, value range is the same as input. if false, value range converted to [0,1] according to image conversion conventions
        # preserve_dtype: if False, output is float as generated during interpolation. if True, output will be converted to input's dtype

        spline = self.splines[spline_i]
        return _warp_2D_img(
            img2D,
            spline,
            self.width,
            self.length,
            scale_factor,
            mirror,
            interpolation_order,
            preserve_range,
            preserve_dtype,
        )

    def warp_3D_img(
        self,
        img3D: NDArray,
        scale_factor: float | tuple[float, float] = 1,
        mirror: bool = False,
        interpolation_order: Literal[0, 1, 2, 3, 4, 5] = 1,
        preserve_range: bool = True,
        preserve_dtype: bool = True,
    ) -> NDArray:
        """Warp full 3D image using generated all generated splines"""
        if len(self.splines) != len(img3D):
            raise ValueError(
                f"Incompatible number of planes in 3D image: {len(img3D)} planes and {len(self.splines)} stored splines"
            )
        return _warp_3D_img(
            img3D,
            self.splines,
            self.width,
            self.length,
            scale_factor,
            mirror,
            interpolation_order,
            preserve_range,
            preserve_dtype,
        )

    def rescaled_3D_img(
        self,
        img3D: NDArray,
        scale_factor: float = 1,
        spacing: tuple[float, float, float] = DEFAULT_SPACING,
        normalise_spacing: bool = True,
        mirror: bool = False,
        interpolation_order: Literal[0, 1, 2, 3, 4, 5] = 1,
        preserve_range: bool = True,
        preserve_dtype: bool = True,
        return_final_spacing: bool = False,
    ) -> NDArray | tuple(NDArray, tuple[float, float, float]):
        """Warp and rescale raw 3D image according to spacing such that resultant spacing is the same in each axis

        spacing is set by microscope acquisition settings and can be found in (raw) image metadata post-experiment
        default spacing is (2.0, 0.325, 0.325), but note, this is specific for each experiment type

        if normalise_spacing is True, spacing is normalised by 'spacing/spacing.min()', keeping pixel length in the respective spacing.min() dimension the same size as provided image. Resultant final_spacing will be spacing.min() / scale_factor in each dimension
        if normalise_spacing is False, pixel lengths in resultant image will be 1/scale_factor units of the underlying physical unit (e.g. micrometers), and so resultant final_spacing will be 1/scale_factor in each dimension
        """
        # scale_factor: how many pixels in straightened image correspond to one pixel in raw
        # mirror: whether to flip left/right relative to raw. original MATLAB code flipped,
        # so what is "left" on the raw image became "right" on the straightened
        # spacing: metadata from microscope acquisition mode - how moving by one pixel in each dimension relates to physical measurements
        # interpolation_order:
        # 0: Nearest-neighbor (e.g. use for masks/labels)
        # 1: Bi-linear (default)
        # 2: Bi-quadratic
        # 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic

        # don't preserve dtype to keep float as image will be rescaled in next step, and excessive dtype conversions will introduce rounding errors
        spacing = np.array(spacing)

        final_spacing = spacing.copy()
        final_spacing[1:] /= scale_factor

        warped_img = self.warp_3D_img(
            img3D,
            scale_factor,
            mirror,
            interpolation_order,
            preserve_range,
            preserve_dtype=False,
        )
        if normalise_spacing:
            scale = final_spacing / final_spacing.min()
        else:
            scale = final_spacing
        rescaled = skimage.transform.rescale(
            warped_img,
            scale,
            interpolation_order,
            preserve_range=preserve_range,
        )
        if preserve_dtype:
            rescaled = rescaled.astype(img3D.dtype)

        if return_final_spacing:
            final_spacing = final_spacing / scale
            return rescaled, final_spacing
        else:
            return rescaled


def validate_mask(mask: NDArray) -> None:
    """
    provided mask should be:
        - 2D or 3D
        - not empty
        - contain a single object (connected component) on each 2D plane.
    """
    if mask.ndim == 1:
        raise ValueError("Provided mask is 1D, but should either be 2D or 3D")
    elif mask.ndim == 2:
        mask = mask[np.newaxis, ...]
    elif mask.ndim > 3:
        raise ValueError(
            f"Provided mask has {mask.ndim} dimensions, but should either be 2D or 3D"
        )

    if np.sum(mask) == 0:
        raise ValueError("Mask is empty so cannot be computationally straightened")

    for i, plane in enumerate(mask):
        num_labels, _ = cv2.connectedComponents(img_as_ubyte(plane), connectivity=4)
        if num_labels > 2:
            raise ValueError(
                f"2D Plane {i} of provided mask contains more than 1 connected component"
            )


####################
# midline extraction
####################


def extract_midline(
    mask2D: NDArray[np.bool_],
    ridge_responce_thresh: float = 90,
    return_dt: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Returns a topologically sorted list of pixels making up the midline of the object

    mask2D: 2D mask of a single object
    dt_percent_thresh: % threshold on ridge response. Lower values retain weaker ridge-pixels
    return_dt: whether to return the calculated distance transform of the mask
    """
    mat, dt = _medial_axis_transform(
        mask2D,
        percentile=ridge_responce_thresh,
        return_distance_transform=True,
    )
    midline = _extract_midline(mat, dt, combined_coord_array=True)
    if return_dt:
        return midline, dt
    else:
        return midline


def _detect_ridges(gray: NDArray, sigma: float = 1) -> tuple[NDArray, NDArray]:
    """Finds ridge points in a grayscale image. Copied from here : https://stackoverflow.com/questions/48727914/how-to-use-ridge-detection-filter-in-opencv"""
    H_elems = hessian_matrix(
        gray,
        sigma=sigma,
        order="rc",
        use_gaussian_derivatives=False,
    )
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def _medial_axis_transform(
    mask2D: NDArray[np.bool_],
    percentile: float = 90,
    ridge_detector: str = "scikit",
    return_distance_transform: bool = False,
) -> NDArray[np.bool_] | tuple[NDArray[np.bool_], NDArray[np.float_]]:
    """medial axis transform of a mask

    MAT calculated by finding pixels where the gradient of the mask's distance transform is discontinuous and thinning the resultant mask to 1 pixel width
    percentile (0<percentile<100) controls sensitivity for discontinuity detection by defining a percentile threshold, above which values are kept
    e.g. 100-90=10 -> top 10% left over
    algorithm from https://doi.org/10.1016/j.apm.2011.05.001

    only works for single component 2D masks"""
    if mask2D.ndim != 2:
        raise ValueError(
            f"Mask must be a 2D array. Provided mask dimensionality is {mask2D.ndim}"
        )
    mask2D = mask2D.astype(bool)

    mask2D = img_as_ubyte(mask2D)
    distance_transform = cv2.distanceTransform(
        mask2D, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )

    if ridge_detector == "scikit":
        ridge_response, _ = _detect_ridges(distance_transform * -1)
    elif ridge_detector == "opencv":
        ridge_response = RIDGE_DETECTOR.getRidgeFilteredImage(distance_transform * -1)

    # get top_percentile of non-zero ridges
    thresh_ridges = ridge_response > np.percentile(
        ridge_response[ridge_response > 0], percentile
    )

    # thresh_ridges has a lot of noise, so extract the largest component to filter away noise
    main_ridge = _largest_mask_component(thresh_ridges)

    # thin to one pixel thickness
    thinned = cv2.ximgproc.thinning(
        img_as_ubyte(main_ridge),
        thinningType=cv2.ximgproc.THINNING_GUOHALL,
    )

    if return_distance_transform:
        return thinned, distance_transform
    else:
        return thinned


def _largest_mask_component(
    mask: NDArray[np.bool_], connectivity: Literal[4, 8] = 8
) -> NDArray[np.bool_]:
    """return a new mask of the largest component in mask

    connectivity 4: cross-shaped connectivity, i.e. no diagonals
    connectivity 8: square-shaped connectivity, i.e. diagonals included"""
    num_labels, labels = cv2.connectedComponents(
        img_as_ubyte(mask), connectivity=connectivity
    )
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_label = unique_labels[np.argmax(counts[1:]) + 1]
    return labels == largest_label


def _extract_midline(
    midline_image: NDArray[np.bool_],
    distance_transform: NDArray[np.float_],
    combined_coord_array: bool = False,
) -> list[NDArray[np.int_]] | list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    returns a topologically sorted pixel path of the midline
    a midline is the longest pixel path of midline image the extended to the boundary
    conceptually, it can be the longest axis of an oblong object

    combined_coord_array:
        False - np.nonzero style indices, a tuple of 2 (M,) arrays is returned for each midline; intended for indexing into skeleton array
        True - np.argwhere style coords, a single (M, 2) array is returned for each midline
    """
    mask = img_as_ubyte(distance_transform > 0)
    if not np.any(mask):
        raise ValueError("Skeleton is empty")

    # start from 1 as not interested in background
    radius = distance_transform.max()
    midline = _main_midline_path(
        midline_image, connectivity=2, pixel_trim=int(radius * 0.5)
    )
    midline = _extend_midline_to_boundary(midline, distance_transform)
    if combined_coord_array:
        midline = np.c_[midline]
    return midline


def _main_midline_path(
    skeleton_img: NDArray[np.bool_],
    connectivity: Literal[1, 2] = 2,
    pixel_trim: int = 0,
) -> NDArray[np.int_]:
    """topologically sorted main path points, trimmed on each end by pixel_trim

    connectivity 1: + shaped neighbour connectivity; i.e. diagonals excluded
    connectivity 2: square shaped neighbour connectivity; i.e. diagonals included
    """
    graph, nodes = skimage.graph.pixel_graph(
        skeleton_img.astype(bool), connectivity=connectivity
    )
    path = _longest_path(graph)
    if pixel_trim > 0:
        pixel_trim = int(pixel_trim)
        if 2 * pixel_trim < len(path):
            # want to avoid trimming the whole path
            path = path[pixel_trim:-pixel_trim]
    flat_indices = nodes[path]
    shaped_indices = np.unravel_index(flat_indices, shape=skeleton_img.shape)
    return shaped_indices


def _longest_path(graph: sparse.csr_matrix) -> NDArray[np.int_]:
    """returns node order of the longest bfs path in graph"""
    if graph.getnnz() == 0:
        return np.array([])
    bfs_visit_order = csgraph.breadth_first_order(
        graph, i_start=0, directed=False, return_predecessors=False
    )
    furthest_node1 = bfs_visit_order[-1]
    bfs_visit_order, predecessors = csgraph.breadth_first_order(
        graph, i_start=furthest_node1, directed=False, return_predecessors=True
    )
    furthest_node2 = bfs_visit_order[-1]

    path = _bfs_path(predecessors, furthest_node2)
    return path


def _bfs_path(predecessors: NDArray[np.int_], node_j: int) -> NDArray[np.int_]:
    """given bfs predecessors and node_j, find path connecting bfs tree's root->node_j"""
    path = []
    current_node = node_j
    while current_node != -9999:
        path.append(current_node)
        current_node = predecessors[current_node]
    # reverse as path is node_j -> root, but bfs would have been root -> node_j
    path.reverse()
    return np.array(path)


###############
# tip extension
###############


def _extend_midline_to_boundary(
    midline: tuple(NDArray[np.int_], NDArray[np.int_]),
    distance_transform: NDArray[np.float_],
) -> NDArray[np.int_]:
    """
    adds two points to midline on the distance_transform background boundary

    independently for each tip, achieved by projecting a diametric ray, finding where the ray intersects the boundary,
    and finding midpoint of the path along boundary connecting the two intersections
    the midpoints are then added to respective ends of midline
    """
    border_radius = distance_transform.max()
    # border consists of pixels that have distance 1 or sqrt(2) to the background
    # border then effectively has connectivity 2 (or cv2 8)
    border = np.logical_or(
        np.isclose(distance_transform, 1), np.isclose(distance_transform, np.sqrt(2))
    )
    # cv2 connectivity 4 is same as skimage connectivity 1; connectivity of 1 pixel distance i.e cross pattern
    # extract the largest component because border may have 'lonely' bits
    border = _largest_mask_component(border, connectivity=4)
    graph, nodes = skimage.graph.pixel_graph(border.astype(bool), connectivity=1)

    # function with preapplied arguments
    def tip_handler(tip, samples):
        return _handle_tip(border, border_radius, graph, nodes, tip, samples)

    midline = np.c_[midline]
    # if calculate finite derivative from just two points at each tip, would get either vertical, horizontal, or 45deg derivative
    # so calculate finite derivative from the last n points at each midline tip
    # cap num_samples to midline size in case midline is too short
    num_samples = np.min([border_radius, len(midline)])
    num_samples = np.round(num_samples).astype(int)
    # get deriv for numsamples from each end of midline
    sample1, sample2 = midline[1:num_samples], midline[-num_samples:-1]
    tip1, tip2 = midline[[0, -1]]
    new_tip1, new_tip2 = tip_handler(tip1, sample1), tip_handler(tip2, sample2)
    new_midline = np.vstack(
        [points for points in (new_tip1, midline, new_tip2) if points is not None]
    )
    return new_midline


def _handle_tip(
    border: NDArray[np.bool_],
    border_radius: float,
    border_graph: sparse.csr_matrix,
    graph_nodes: list[int],
    origin: tuple[int, int],
    point_samples: NDArray[np.int_],
) -> tuple[int, int] | None:
    """calculates mean vector point_samples -> origin
    projects a perpendicular diameter at origin
    finds where diameter intersects border
    graph seach for path connecting the two intersections
    and returns midpoint of path"""
    parallel_vector = _mean_unit_vector(origin, point_samples)
    normal_vector = np.array([-1, 1]) * parallel_vector[::-1]
    # extend radius so ensure it crosses the border. otherwise radius might barely miss the border
    # left and right radii together form a diameter centered at origin
    left_radius_points = _generate_ray_points(
        origin, normal_vector, border_radius * 1.25, border.shape
    )
    right_radius_points = _generate_ray_points(
        origin, -normal_vector, border_radius * 1.25, border.shape
    )
    # get the two points where the radii cross the border
    # argmax because most of radius will evaluate 0, and when radius intersects border it will evaluate 1
    # index into radius_points
    point_index = np.argmax(border[left_radius_points], axis=0)
    # row,col point
    start_point = np.r_[
        left_radius_points[0][point_index], left_radius_points[1][point_index]
    ]
    # index into border.flatten(), which is what graph_nodes contains
    start_point_flat_index = np.sum(start_point * border.strides // border.itemsize)
    # there may not be any point on the boundary, e.g. if mask goes up to the edge of the image
    try:
        # index of point within graph_nodes gives index into graph itself
        start_node = np.flatnonzero(graph_nodes == start_point_flat_index)[0]
    except IndexError:
        return None

    # repeat above for second point
    point_index = np.argmax(border[right_radius_points], axis=0)
    end_point = np.r_[
        right_radius_points[0][point_index],
        right_radius_points[1][point_index],
    ]
    end_point_flat_index = np.sum(end_point * border.strides // border.itemsize)
    try:
        end_node = np.flatnonzero(graph_nodes == end_point_flat_index)[0]
    except IndexError:
        return None

    visit_order, predecessors = csgraph.breadth_first_order(
        border_graph, start_node, directed=False, return_predecessors=True
    )
    path = _bfs_path(predecessors, end_node)
    # if path traverses more than half the perimeter of border, new tip_point is not going to be good
    if len(path) > len(np.flatnonzero(border)) / 2:
        return None
    tip_node = path[len(path) // 2]
    tip_point = np.unravel_index(graph_nodes[tip_node], shape=border.shape)
    return tip_point


def _generate_ray_points(
    origin: tuple[int, int],
    unit_vector: tuple[float, float],
    ray_length: float,
    clipping_shape: tuple[int, int] | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """return integer rows,cols crossed by ray starting at origin, in direction of unit_vector, of length ray_length

    clipping_shape allows clipping of points that would be outside of grid with that shape
    """
    start = np.round(origin).astype(int).tolist()

    unit_vector = np.array(unit_vector)
    end = origin + unit_vector * ray_length
    end = np.round(end).astype(int).tolist()

    points = skimage.draw.line(*start, *end)
    if clipping_shape is not None:
        rows, cols = points
        n_rows, n_cols = clipping_shape
        allowed_rows = (0 <= rows) & (rows < n_rows)
        allowed_cols = (0 <= cols) & (cols < n_cols)
        allowed_both = allowed_rows & allowed_cols
        points = (rows[allowed_both], cols[allowed_both])

    return points


def _mean_unit_vector(
    main_point: tuple[int, int], point_samples: NDArray[np.int_]
) -> tuple[float, float]:
    """calculates mean unit vector of point_samples -> main_point"""
    finite_derivs = point_samples - main_point
    # normalise finite_derivs so avoid bias towards points far away from main_point
    finite_derivs = finite_derivs / np.linalg.norm(finite_derivs, axis=1)[:, np.newaxis]
    mean_deriv = np.mean(finite_derivs, axis=0)
    # mean not necessarily unit length, so normalise again
    mean_deriv = mean_deriv / np.linalg.norm(mean_deriv)
    return mean_deriv


###############################
# spline and coord manipulation
###############################

# There are three sets of coordinates:
# raw_img xy-coordinates
# straightened_img x'y'-coordinates
# and intermediate uv spline coordinates
# spline uv coordinates are just x'y' coordinates where origin is is shifted down to be vertically centered, instead of being in the top-left corner

# basic idea of the transform is to convert uv coordinates to xy coordinated in raw img_as_ubyte
# u - how far to walk along spline (always +ve)
# v - how far to walk perpendicularly to spline. (+ve and -ve values correspond to which orthogonal to follow)
# so xy = spline(u) + v * orthog


def _from_spline_coords_to_raw(
    spline: BSpline, spline_coords: NDArray[np.float_], mirror: bool = False
) -> NDArray[np.float_]:
    """convert from coordinates in spline domain to coordinates in spline range

    spline_coords col0 is axis perpendicular to spline, col1 is axis parallel to spline
    """
    # left or right orthogonal
    if mirror:
        # right orthogonal
        orthog = np.array([-1, 1])
    else:
        # left orthogonal
        orthog = np.array([1, -1])
    # tuple e.g. if np.nonzero() is used on straightened 2D image
    if isinstance(spline_coords, tuple):
        widths, lengths = spline_coords
    # otherwise will be numpy (N, 2) numpy array, e.g. if np.argwhere() is used on straightened 2D image
    else:
        widths, lengths = spline_coords.T
    origins = spline(lengths)
    # parallel vectors
    derivs = spline.derivative(1)(lengths)
    # orthogonal vectors
    normals = np.roll(derivs, shift=1, axis=1) * orthog
    # normalised orthogonal vectors
    normals = normals / np.linalg.norm(normals, axis=1)[..., np.newaxis]
    img_yxs = origins + normals * widths[..., np.newaxis]
    return img_yxs


def _from_grid_coords_to_spline(
    grid_coords: NDArray[np.int_],
    length: float,
    width: float,
    scale_factor: float | tuple[float, float] = np.array([1, 1]),
) -> NDArray[np.float_]:
    """coordinates are scaled according to scale_factor, and translated such that spline's length axis is place in middle of grid axis0

    scale_factor: how many units on grid equal one unit in spline domain"""
    spline_coords = grid_coords / scale_factor
    # spline widths are centred around 0, grid widths are centred around worm_width / 2
    offset = np.array([width / 2, 0])
    spline_coords = spline_coords - offset
    return spline_coords


def _fit_spline(
    points: NDArray[np.float_],
    parametrisation: list[float],
    smoothing_factor: float,
) -> BSpline:
    """fit a spline to the points and return a scipy BSpline object"""
    # number of points needs to be greater than the degree of spline, which is 3
    if len(points) < 4:
        return None
    ys, xs = points.T
    try:
        tck, u = splprep([ys, xs], u=parametrisation, s=smoothing_factor)
    except Exception as e:
        print(
            "Something went wrong in spline fitting, returning None and resuming execution."
        )
        import traceback

        print("Error traceback:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None
    t, c, k = tck
    # splprep returns "c" in wrong shape, unlike splrep
    c = np.asarray(c).T
    # splprep values guaranteed to be correct, so construct fast without checks
    spline = BSpline.construct_fast(t, c, k)
    return spline


def _warped_coords(
    width: float,
    length: float,
    spline: BSpline,
    scale_factor: float | tuple[float, float] = 1,
    mirror: bool = False,
) -> NDArray[np.float_]:
    """given dimensions and a spline, creates a [2,M,N] array, where M,N are warped image dimensions, and axis0 YX float coordinates into original image
    coordinates are float because values will be interpolated between integer coordinates in original image
    """

    def coord_map(grid_indices: NDArray[np.int_]) -> NDArray[np.float_]:
        # current implementation of skimage.transform.warp_coords has a bug where row,col are swapped relative to what documentation says
        # so have to reverse axis=1 so that the internal code calls it's own dependencies correctly
        # but have to reverse axis=0 afterwards because the internal code assumes row,col when reshaping, when actually it called dependencies with col,row
        spline_coords = _from_grid_coords_to_spline(
            grid_indices[:, ::-1], length, width, scale_factor
        )
        raw_img_coords = _from_spline_coords_to_raw(
            spline, spline_coords, mirror=mirror
        )
        return raw_img_coords

    warped_shape = _warped_shape(width, length, scale_factor)
    # TODO: remove axis reversal when bug is fixed
    warped_coords = skimage.transform.warp_coords(
        coord_map=coord_map,
        shape=warped_shape,
    )
    warped_coords = warped_coords[::-1, ...]
    return warped_coords


def _refine_spline(
    smooth_spline: BSpline,
    extrapolation_range: float = 0.01,
) -> tuple[BSpline, float, list[float]]:
    """given a smooth spline, refine it such that 1unit in spline domain equals 1unit in spline range.
    slightly extrapolates beyond ends of spline by extrapolation_range (as fraction of total length) to be sure to cover full real length.

    does not apply further smoothing"""
    if smooth_spline is None:
        return None
    # refines spline by fitting on parametrisation that respects inter-pixel distance
    points = smooth_spline(
        np.linspace(-extrapolation_range, 1 + extrapolation_range, 1000)
    )
    bb_dists = np.linalg.norm(points[:-1] - points[1:], axis=1)
    parametrisation = np.r_[0, np.cumsum(bb_dists)]
    spline_length = parametrisation[-1]

    # don't smooth because points already smooth
    refined_spline = _fit_spline(points, parametrisation, smoothing_factor=0)

    return refined_spline, spline_length, parametrisation


def _align_splines(
    img3D: NDArray,
    worm_width: float,
    worm_length: float,
    splines: list[BSpline],
    spline_parametrisations: list[list[float]],
) -> tuple[list[BSpline], float]:
    """uses splines to warp a 3D image to use for aligning splines along spline length.

    i.e. all of unaligned spline domains all start at 0. each of aligned splines' domains will start wherever necessary so that resultant warped image planes are aligned with each other

    Alignment along spline width is implied by accurate midline generation in original spline fitting
    """
    # fits new splines with offsets on parametrisation such that they are aligned together
    warped_img = _warp_3D_img(img3D, splines, worm_width, worm_length)
    feature_img = np.stack(
        [skimage.filters.farid_v(plane) for plane in warped_img], axis=0
    )
    feature_img = feature_img.mean(axis=1)
    alignment_offsets = [
        skimage.registration.phase_cross_correlation(
            ref_img, mov_img, upsample_factor=20, normalization=None
        )[0]
        for ref_img, mov_img in zip(feature_img[:-1], feature_img[1:])
    ]
    # alignments relative to 0th plane, which has offset of 0 to itself
    alignment_offsets = np.concatenate([[0], *alignment_offsets])
    cuml_aln_offsets = np.cumsum(alignment_offsets)
    aligned_parametrisations = [
        prm + offset for prm, offset in zip(spline_parametrisations, cuml_aln_offsets)
    ]
    # change alignment so that no splines have negative length parametrisations
    min_parametrisation = min(prm[0] for prm in aligned_parametrisations)
    aligned_parametrisations = [
        prm - min_parametrisation for prm in aligned_parametrisations
    ]
    coords = [spline(prm) for spline, prm in zip(splines, spline_parametrisations)]
    aligned_splines = [
        _fit_spline(bb, aln_prm, smoothing_factor=0)
        for bb, aln_prm in zip(coords, aligned_parametrisations)
    ]
    # before, length was the length of the longest spline domain, but now want the length of the union of all spline domains
    # since the union of domains starts at 0, the length of the union is simply the largest number reached by any spline
    length = max(prm[-1] for prm in aligned_parametrisations)
    return aligned_splines, length


def _handle_flips(midlines: list[NDArray | None]) -> list[NDArray | None]:
    """
    Ensure consistent head/tail orientation across a list of midlines.

    Iterates over the midlines in order, comparing each midline's endpoints to
    the previous one. If the endpoints suggest a head/tail swap (i.e. the
    flipped assignment has a smaller total Euclidean distance), the midline is
    reversed before being added to the output.

    Parameters:
        midlines (list[np.ndarray or None]): Ordered list of midline coordinate
            arrays (each of shape ``(N, 2)``). ``None`` entries for empty planes
            are passed through unchanged.

    Returns:
        list[np.ndarray or None]: Midlines with consistent orientation.
    """
    # initialise
    for ml in midlines:
        if ml is not None:
            current_tips = ml[[0, -1]]
            break
    final_midlines = []
    for ml in midlines:
        if ml is None:
            final_midlines.append(None)
            continue
        next_tips = ml[[0, -1]]
        if _are_tips_flipped(current_tips, next_tips):
            new_ml = np.flip(ml, axis=0)
        else:
            new_ml = ml
        final_midlines.append(new_ml)
        current_tips = new_ml[[0, -1]]
    return final_midlines


def _are_tips_flipped(tip_pair1: NDArray[np.int_], tip_pair2: NDArray[np.int_]) -> bool:
    """Should tip assignment be flipped based on euclidian distance penalty"""
    # pair_wise_dists[i, j] distance between tip_pair1[i] and tip_pair2[j]
    pair_wise_dists = distance.cdist(tip_pair1, tip_pair2, metric="euclidean")
    unflipped_dist = np.sum(pair_wise_dists[[0, 1], [0, 1]])
    flipped_dist = np.sum(pair_wise_dists[[0, 1], [1, 0]])
    return flipped_dist < unflipped_dist


###############
# image warping
###############


def _warp_2D_img(
    img2D: NDArray,
    spline: BSpline,
    width: float,
    length: float,
    scale_factor: float | tuple[float, float],
    mirror: bool = False,
    interpolation_order: Literal[0, 1, 2, 3, 4, 5] = 1,
    preserve_range: bool = False,
    preserve_dtype: bool = False,
) -> NDArray:
    """
    Warp a single 2D image plane using spline-based coordinate mapping.

    Computes the inverse coordinate map via :func:`_warped_coords` and applies
    ``skimage.transform.warp``. Returns a zero-filled array when ``spline`` is
    ``None``.

    Parameters:
        img2D (NDArray): 2D input image plane.
        spline (BSpline or None): Fitted spline defining the midline mapping.
        width (float): Output image width (pixels across the midline).
        length (float): Output image length (pixels along the midline).
        scale_factor (float or tuple[float, float]): Scale factor(s) applied to
            the output grid.
        mirror (bool, optional): If ``True``, flip the midline direction.
            (default: False)
        interpolation_order (int, optional): Spline interpolation order for
            ``skimage.transform.warp`` (0–5). (default: 1)
        preserve_range (bool, optional): Passed to ``skimage.transform.warp``.
            (default: False)
        preserve_dtype (bool, optional): If ``True``, cast the output back to
            ``img2D.dtype``. (default: False)

    Returns:
        NDArray: Warped 2D image of shape ``(width * scale, length * scale)``.
    """
    if spline is None:
        shape = _warped_shape(width, length, scale_factor)
        return np.zeros(shape)
    warped_coords = _warped_coords(width, length, spline, scale_factor, mirror)
    warped_img = skimage.transform.warp(
        img2D,
        warped_coords,
        order=interpolation_order,
        preserve_range=preserve_range,
    )
    if preserve_dtype:
        warped_img = warped_img.astype(img2D.dtype)
    return warped_img


def _warp_3D_img(
    img3D: NDArray,
    splines: BSpline,
    width: float,
    length: float,
    scale_factor: float | tuple[float, float] = 1,
    mirror: bool = False,
    interpolation_order: Literal[0, 1, 2, 3, 4, 5] = 1,
    preserve_range: bool = False,
    preserve_dtype: bool = False,
) -> NDArray:
    """
    Apply :func:`_warp_2D_img` to each plane of a 3D image stack.

    Parameters:
        img3D (NDArray): 3D input image stack of shape ``(N, H, W)``.
        splines (list[BSpline or None]): One spline per plane; must have the same
            length as ``img3D``.
        width (float): Output image width (pixels across the midline).
        length (float): Output image length (pixels along the midline).
        scale_factor (float or tuple[float, float], optional): Scale factor(s)
            applied to the output grid. (default: 1)
        mirror (bool, optional): If ``True``, flip the midline direction.
            (default: False)
        interpolation_order (int, optional): Spline interpolation order (0–5).
            (default: 1)
        preserve_range (bool, optional): Passed to ``skimage.transform.warp``.
            (default: False)
        preserve_dtype (bool, optional): If ``True``, cast each output plane back
            to the input plane's dtype. (default: False)

    Returns:
        NDArray: Warped stack of shape ``(N, width * scale, length * scale)``.
    """
    warped_img = [
        _warp_2D_img(
            plane,
            spline,
            width,
            length,
            scale_factor,
            mirror,
            interpolation_order,
            preserve_range,
            preserve_dtype,
        )
        for plane, spline in zip(img3D, splines)
    ]
    warped_img = np.stack(warped_img, axis=0)
    return warped_img


def _warped_shape(
    width: float,
    length: float,
    scale_factor: float | tuple[float, float] = 1,
) -> tuple[int, int]:
    """2D grid shape that will contain the full object, scaled by scale_factor

    currently, no way to add margins has been implemented"""
    shape = np.array([width, length]) * scale_factor
    shape = np.ceil(shape).astype(int)
    return shape
