import numpy as np
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.isdir("out"):
    os.makedirs("out")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Slicer")
    parser.add_argument(
        "depthmap",
        type=str,
        nargs="?",
        default="datasets/scan 1/surface.csv",
        help="Path to depthmap CSV"
    )
    return parser.parse_args()

args = parse_args()
DEPTHMAP_FILENAME = args.depthmap
ROUTES_JSON_OUTPUT_FILEPATH = "out/routes.json"
ENABLE_PLOTS = True

pixel_size = 1.125 # Uniform pixel size(separate xy spans not implemented)
DISK_RADIUS_MM = 75.0 # Grinder disk radius
REFERENCE_FORCE = 1.5*9.8
REFERENCE_PRESSURE = REFERENCE_FORCE / (np.pi * DISK_RADIUS_MM/1000 * DISK_RADIUS_MM/1000) #
MIN_ACTUATOR_FORCE = 4.9 # N
MAX_ACTUATOR_FORCE = 49 # N
# DENSITY PARAMETERS
R_MM = DISK_RADIUS_MM/4 # Small disk radius
SAMPLING_DISTANCE = R_MM*np.sqrt(4/3)
MIN_POINTS = 16
SMALL_DISK_INLIER_THRESHOLD = 1.
COLLISION_THRESHOLD_MM = 3.0
PENETRATION_VOLUME_THRESHOLD = 5*5*5. # mm^3
MIN_NEW_WORKED_AREA = 1/5
# WORKED_POINTS_OVERLAP_THRESHOLD = 75/100 #WORKED_POINTS_OVERLAP_THRESHOLD OVERLAP[%]
CONNECTIVITY_RADIUS = 2*DISK_RADIUS_MM*np.sqrt(1**2 - (8/10)**2)#2.*DISK_RADIUS_MM#<=2*DISK_RADIUS_MM
SECTOR_COUNT = 6
K_NEAREST_NEIGHBORS = 16
# Optimizer parameters
OPTIMIZE_END_NODE = True
DUMMY_COST = 0.0
RELATIVE_GAP_THRESHOLD = 0.00 # 0.05
NORMAL_CHANGE_LAMBDA = 0.2
RELAXED_PROBLEM_SIZE_THRESHOLD = 0 # Size under which less strict constraints are imposed, yields for better optimality
DISCOUNT_FACTOR = 1/(1/4)
BACK_EDGE_COST = 8.0 / 1.0
# Route parameters
MINIMUM_ROUTE_LENGTH = 4
MAX_HEIGHT_CHANGE_MM = 3.0
MAX_NORMAL_CHANGE_DEGREE = 3.
MAX_NORMAL_CHANGE_DEGREE_OVER_DISTANCE = MAX_NORMAL_CHANGE_DEGREE / (SAMPLING_DISTANCE) # Change over distance
MIN_NODES_FOR_BOUNDARY = 3


import time
script_start_time = time.time()

import pandas as pd
import open3d as o3d
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge
from pulp import (
    value,
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpBinary,
    LpContinuous,
    HiGHS,
    SCIP_PY,
    PULP_CBC_CMD,
    LpStatusOptimal
)
import networkx as nx
from tqdm import tqdm
from pyransac3d import Plane

def timer(func):
    """Timing decorator"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

## PLOTTING UTILS --------------------
import pyvista as pv

def create_disks_as_points_and_lines(centers, normals, inner_radius, outer_radius, c_res=24):
    """
    Create disks visualization using points and lines instead of full meshes
    - Center point
    - Outer perimeter points  
    - Normal vector line
    """
    
    all_points = []
    all_lines = []
    point_count = 0
    if inner_radius is not np.ndarray:
        inner_radius = np.full(centers.shape[0],inner_radius)
    if outer_radius is not np.ndarray:
        outer_radius = np.full(centers.shape[0],outer_radius)
    for center, normal, inner_r, outer_r in zip(centers, normals, inner_radius, outer_radius):
        center = np.array(center)
        #normal = np.array(normal) / np.linalg.norm(normal)
        
        # Add center point
        all_points.append(center)
        center_idx = point_count
        point_count += 1
        
        # Create outer perimeter points
        # Find two orthogonal vectors to the normal
        if abs(normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Generate perimeter points
        angles = np.linspace(0, 2*np.pi, c_res, endpoint=False)
        if not np.isclose(inner_r,0.0):
            perimeter_points = []
            for angle in angles:
                point = center + inner_r * (np.cos(angle) * u + np.sin(angle) * v)
                all_points.append(point)
                perimeter_points.append(point_count)
                point_count += 1
            # Add lines for perimeter (connect perimeter points in circle)
            for i in range(len(perimeter_points)):
                next_i = (i + 1) % len(perimeter_points)
                all_lines.append([2, perimeter_points[i], perimeter_points[next_i]])  # 2 points per line

        perimeter_points = []
        for angle in angles:
            point = center + outer_r * (np.cos(angle) * u + np.sin(angle) * v)
            all_points.append(point)
            perimeter_points.append(point_count)
            point_count += 1
        
        # Add lines for perimeter (connect perimeter points in circle)
        for i in range(len(perimeter_points)):
            next_i = (i + 1) % len(perimeter_points)
            all_lines.append([2, perimeter_points[i], perimeter_points[next_i]])  # 2 points per line
        
        # Add normal vector line
        normal_end = center + normal * (outer_r * 1.)  # Scale normal for visibility
        all_points.append(normal_end)
        normal_end_idx = point_count
        point_count += 1
        all_lines.append([2, center_idx, normal_end_idx])
    
    if not all_points:
        return pv.PolyData()
    
    # Combine all points
    points = np.array(all_points)
    
    # Combine all lines
    if all_lines:
        lines = np.array([item for line in all_lines for item in line])
    else:
        lines = np.array([])
    
    # Create polydata
    polydata = pv.PolyData(points, lines=lines)
    return polydata

def add_disks_as_points_and_lines(plotter, centers, normals, inner_radius, outer_radius, 
                                 c_res=24, color="red", opacity=0.6, point_size=5):
    """
    Fast visualization of disks as points and lines
    """
    polydata = create_disks_as_points_and_lines(centers, normals, inner_radius, outer_radius, c_res)
    
    if polydata.n_points > 0:
        plotter.add_mesh(polydata, color=color, opacity=opacity, line_width=2, point_size=point_size)
    
    return polydata

def plot_and_animate(point_cloud, disks, path, mp4_filename, constrained_edges):
    plotter = pv.Plotter(off_screen=False)
    z_coords = point_cloud[:, 2]
    z_min, z_max = np.percentile(z_coords, [5, 95])
    z_coords = z_max - point_cloud[:, 2] + z_min
    plotter.add_points(point_cloud, point_size=2, render_points_as_spheres=False, opacity=0.3, scalars=z_coords, cmap="viridis",clim=[z_min, z_max])
    plotter.add_points(disks[path[0][0]][:3], color="green", point_size=10,
                    render_points_as_spheres=False)
    plotter.add_points(disks[path[-1][-1]][:3], color="red", point_size=10,
                    render_points_as_spheres=False)
    # Initial moving disk at start position
    disk_obj = pv.Disc(center=disks[path[0][0],:3], inner=0.0, outer=DISK_RADIUS_MM, r_res=1, c_res=24, normal=disks[path[0][0],3:6])
    plotter.add_mesh(
        disk_obj,
        color="brown", opacity=1.0
    )

    _disks = np.array([disks[node] for node in cc])
    polydata = add_disks_as_points_and_lines(
        plotter, _disks[:, :3], _disks[:, 3:6], R_MM, DISK_RADIUS_MM,
        c_res=24, color="black", opacity=0.5
    )
    if constrained_edges:
        import matplotlib.pyplot as plt

        # Get a colormap with distinct colors
        colors = plt.cm.Set1(np.linspace(0, 1, len(constrained_edges)))

        for idx, (_path, color) in enumerate(zip(constrained_edges, colors)):
            # Convert color from matplotlib format to RGB tuple
            rgb_color = tuple(color[:3])  # Remove alpha channel
            
            plotter.add_lines(
                np.array([(disks[i][:3], disks[j][:3]) for i, j in _path]).reshape(-1, 3), 
                color=rgb_color, 
                width=3
            )

    # Setup view
    plotter.view_yx()
    plotter.show_grid()
    plotter.camera.view_up = [0, -1, 0]
    plotter.camera.elevation = -30
    plotter.zoom_camera(1.5)
    plotter.show(auto_close=False, interactive_update=True, full_screen=True)

    # Open GIF file
    plotter.open_movie(mp4_filename, framerate=24, quality=7)
    # Animate path
    n_steps = 5  # Number of interpolation steps per segment

    for i, (n1, n2) in enumerate(path):
        start_point = disks[n1][:3]
        end_point = disks[n2][:3]
        
        # Interpolate between start and end points
        for step in range(n_steps):
            t = step / (n_steps - 1)  # Interpolation parameter (0 to 1)
            current_pos = start_point + t * (end_point - start_point)
            
            # Update disk position
            new_center = current_pos
            new_normal = disks[n2, 3:6]  # Use target normal
            new_disc = pv.Disc(center=new_center, inner=0.0,
                            outer=DISK_RADIUS_MM, r_res=1, c_res=24,
                            normal=new_normal)
            disk_obj.shallow_copy(new_disc)
            plotter.write_frame()
        
        # Add the line segment after completion (no animation)
        segment = np.array([start_point, end_point])
        plotter.add_lines(segment, color="red", width=3)
        # if i%10==0:
        #     plotter.save_graphic(f"{mp4_filename[:-4]}_{i}.pdf")

    plotter.close()
## END PLOTTING UTILS ----------------


# Function to plot the 3D surface

def read_depthmap(filename):
    """
    Reads a surface with the following format
    x_span, y_span
    first vertical scan
    second vertical scan
    """
    spans = pd.read_csv(filename, nrows=1,header=None).values
    zz = pd.read_csv(filename, skiprows=1,header=None).values
    x = np.arange(zz.shape[0])*spans[0,0]
    y = np.arange(zz.shape[1])*spans[0,1]
    xx,yy = np.meshgrid(x,y)

    def median_filter(data, size=3):
        """Vectorized median filter using stride tricks"""
        from numpy.lib.stride_tricks import sliding_window_view
        
        pad_size = size // 2
        padded = np.pad(data, pad_size, mode='edge')
        windows = sliding_window_view(padded, (size, size))
        return np.median(windows, axis=(-2, -1))
    def uniform_filter(data, size=3):
        """Average filter using numpy's convolve"""
        from numpy.lib.stride_tricks import sliding_window_view

        
        kernel = np.ones((size, size)) / (size * size)
        pad_size = size // 2
        padded = np.pad(data, pad_size, mode='edge')
        windows = sliding_window_view(padded, (size, size))
        return np.sum(windows * kernel, axis=(-2, -1))
    zz = median_filter(zz, size = 3)
    zz = uniform_filter(zz, size = 3)
    return xx,yy,zz

# @timer
def depth_to_pointcloud(depth_map, pixel_size=1.125):
    H, W = depth_map.shape
    x_coords, y_coords = np.mgrid[0:H, 0:W]
    x = x_coords * pixel_size
    y = y_coords * pixel_size
    z = depth_map
    return np.stack([x, y, z], axis=-1)

def resample_points(points, max_points=1000, random_state=42):
    """Resample points to a maximum number for efficient processing"""
    np.random.seed(random_state)
    if len(points) <= max_points:
        return points
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]

def fit_plane_ransac_implicit(points_3d, max_trials=100, approximate_residual_threshold = SMALL_DISK_INLIER_THRESHOLD, max_population = 10000):
    """
    RANSAC estimator for robust plane fitting. The estimator fits the equation where [a,b,c] is an unit vector:
    
    ax + by + cz + d = 0
    ax + by + cz = -d

    The above expression can be divided by d as long as d!=0(plane not passing through the origin), yielding the unambigous equation:
    a/d x + b/d y + c/d z = -1
    a'x + b'y + c'z = -1
    Since [a,b,c] is an unit vector:
    1/|d| = norm([a',b',c']), hence |d| = 1/norm([a',b',c'])

    The unit normal [a,b,c] follows the upward facing convention, hence if c'>0 and c>0 then d>0:
    d = 1/norm([a',b',c'])
    if c'<0 and c>0 then d<0:
    d = -1/norm([a',b',c'])
    """
    if len(points_3d) < 3:
        return None
    
    if max_population is not None and len(points_3d) > max_population:
        points_3d = resample_points(points_3d, max_population)
    plane = Plane()
    params, inliers=plane.fit(points_3d,SMALL_DISK_INLIER_THRESHOLD,minPoints=4,maxIteration=max_trials)
    if params[-1] < 0:
        return -np.array(params)
    return np.array(params)
    

def get_column(surface_point_cloud, x0, n, pixel_size=pixel_size, column_height=2*DISK_RADIUS_MM, asymmetric_ratio=1.0, radius=DISK_RADIUS_MM):
    n = np.array(n)
    n = n / np.linalg.norm(n)
    x1 = -n * column_height * (asymmetric_ratio / (asymmetric_ratio + 1.0)) + x0
    x2 = n * column_height * (1.0 / (asymmetric_ratio + 1.0)) + x0

    # Map from mm to pixel
    H, W, _ = surface_point_cloud.shape
    min_x_mm, min_y_mm = surface_point_cloud[0,0,0:2]
    max_x_mm, max_y_mm = surface_point_cloud[H-1,W-1,0:2]
    assert pixel_size == (max_y_mm - min_y_mm)/(W-1)
    assert pixel_size == (max_x_mm - min_x_mm)/(H-1)
    def pixel(mm, min_mm, max_mm, size):
        return int((mm - min_mm)/(max_mm - min_mm)*(size-1))
    A = np.array([pixel(x1[0], min_x_mm, max_x_mm, H),pixel(x1[1], min_y_mm, max_y_mm, W)])
    B = np.array([pixel(x2[0], min_x_mm, max_x_mm, H),pixel(x2[1], min_y_mm, max_y_mm, W)])
    R_pix = int(radius / pixel_size)

    min_x = max(0, min(A[0], B[0]) - R_pix)
    max_x = min(max(A[0], B[0]) + R_pix, H)
    min_y = max(0, min(A[1], B[1]) - R_pix)
    max_y = min(max(A[1], B[1]) + R_pix, W)

    

    points = surface_point_cloud[min_x:max_x, min_y:max_y]
    if not (max_x > min_x and max_y > min_y):
        return None, None, None, None

    diffs = points - x0 #(N,M,3)
    dot_products = np.matmul(diffs,n)#np.einsum("ijk,k -> ij", diffs, n) #(N,M)
    projected_diffs = diffs - np.einsum("ij,a -> ija", dot_products, n) #(N,M,3)
    mask = np.einsum('ijk,ijk->ij', projected_diffs, projected_diffs) <= radius**2
    return (min_x,max_x),(min_y,max_y), points, mask


def compute_surface_area_from_grad(gradients, valid_mask, threshold=0.0, pixel_size=pixel_size):
    """
    Use gradients to estimate surface area.
    """
    
    # Compute gradients (partial derivatives)
    dz_dx, dz_dy = gradients
    
    # Compute surface element: sqrt(1 + (dz/dx)² + (dz/dy)²)
    surface_element = np.sqrt(1 + dz_dx[valid_mask]**2 + dz_dy[valid_mask]**2)
    
    # Integrate over valid region
    area_estimate = np.sum(surface_element) * (pixel_size * pixel_size)
    
    return area_estimate

def worked_points(points_mm, disk_center_mm, R_mm, normal_mm, pixel_size=1.125, force_threshold=None, max_pressure=None):
    """
    Identify points that would be "worked" (compressed) on a disk surface based on a force threshold model.
    
    This function simulates a compression process where points within a disk area are compressed
    until the total force reaches a specified threshold. It returns the 2D indices of points
    that would be affected by this compression.
    
    Parameters:
    -----------
    points_mm : numpy.ndarray
        3D array of shape (H, W, 3) containing 3D point coordinates in millimeters.
        Each pixel contains [x, y, z] coordinates.
    disk_center_mm : array-like
        3D coordinates [x, y, z] of the disk center in millimeters.
    R_mm : float
        Radius of the disk in millimeters.
    normal_mm : array-like
        3D normal vector of the disk surface in millimeters.
    pixel_size : float, optional
        Size of each pixel in millimeters (default: 1.125).
    force_threshold : float, optional
        Force threshold in arbitrary units. When total compression force reaches
        this value, compression stops (default: None).
    max_pressure : float, optional
        Pressure threshold. When pressure drops below this value, compression stops (default: None).
        
    Returns:
    --------
    tuple or None
        If valid points are found: (rows, cols) where:
            - rows: array of row indices of worked points
            - cols: array of column indices of worked points
        If no valid region or no points found: None
    """
    
    magic_number = 1000
    # Validate input - exactly one threshold must be provided
    if (force_threshold is None) == (max_pressure is None):
        raise ValueError("Exactly one of force_threshold or max_pressure must be provided (not both None or both not None)")
    def debug_print(**kwargs):
        for key, value in kwargs.items():
            print(f"{key}: {value}")

    # Get image dimensions
    H, W = points_mm.shape[:2]
    
    # Normalize the normal vector to ensure it's a unit vector
    unit_normal_mm = normal_mm / np.linalg.norm(normal_mm)
    
    # Use get_column to extract points within the disk column
    # Using a large column_height to ensure we capture all relevant points
    x_interval, y_interval, bb_points, points_mask = get_column(points_mm, disk_center_mm, unit_normal_mm, column_height=7*R_mm, asymmetric_ratio=6.0, radius=R_mm)
    if bb_points is None:
        return None, None, None, None, None
    min_x, max_x = x_interval
    min_y, max_y = y_interval
    
    # Calculate perpendicular distances from each point to the disk plane
    # Positive distance = point is above the plane, Negative = point is below
    diffs = bb_points - disk_center_mm #(N,M,3)
    rvec = np.cross(unit_normal_mm, [0,0,1])
    if np.isclose(np.linalg.norm(rvec), 0.0):
        rotation_matrix = np.eye(3)
    else:
        rvec = rvec / np.linalg.norm(rvec) * np.acos(unit_normal_mm[2])
        # Convert rvec to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    diffs = np.einsum("ab,ijb -> ija", rotation_matrix, diffs)
    distances = diffs[:,:,2]  # distances along z-axis after rotation
    
    # Simulation parameters
    if max_pressure is not None:
        max_displacement = max_pressure * (pixel_size/1000 * pixel_size/1000) * magic_number
    else:
        max_displacement = 2.5
    min_compression = np.min(distances[points_mask])
    
    def evaluate(displacement):
        worked_mask = points_mask & (distances <= displacement)
        if np.count_nonzero(worked_mask) == 0:
            return None, None
        local_force = (displacement - distances[worked_mask]) / magic_number
        force_val = np.sum(local_force)
        return force_val
    
    # Initialize optimal values
    opt_compression = None
    force = None
    feasible = False
    
    # Choose search strategy based on which threshold is provided
    if force_threshold is not None:
        tol = 1e-1
        max_iter = 20
        # Binary search on force
        low, high = min_compression, max_displacement
        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            force_val = evaluate(mid)
            if force_val is not None and force_val >= force_threshold:
                opt_compression = mid
                force = force_val
                feasible = True
                high = mid  # search for smaller compression
            else:
                low = mid
            
            if high - low < tol:
                break
                
    elif max_pressure is not None:
        force_val = evaluate(max_displacement) #total force at maximum local pressure
        feasible = MIN_ACTUATOR_FORCE <= force_val <= MAX_ACTUATOR_FORCE
        if MIN_ACTUATOR_FORCE <= force_val:
            if force_val <= MAX_ACTUATOR_FORCE:
                force = force_val
                opt_compression = max_displacement
                feasible = True
            else:
                feasible = False
                tol = 1e-1
                max_iter = 20
                # Binary search on force
                low, high = min_compression, max_displacement
                for _ in range(max_iter):
                    mid = 0.5 * (low + high)
                    force_val = evaluate(mid)
                    if force_val is not None and force_val >= MAX_ACTUATOR_FORCE:
                        opt_compression = mid
                        force = force_val
                        feasible = True
                        high = mid  # search for smaller compression
                    else:
                        low = mid
                    
                    if high - low < tol:
                        break
        else:
            feasible = False
    
    if not feasible:
        return None, None, None, None, None
    
    # Recompute final values
    force = evaluate(opt_compression)
    
    # Find points that would be worked based on optimal compression
    effectively_worked_mask = points_mask & (distances <= opt_compression)
    worked_points_coords = bb_points[effectively_worked_mask]

    # Vectorized conversion of 3D coordinates to 2D pixel indices
    if len(worked_points_coords) > 0:
        # Vectorized coordinate conversion
        pixel_coords = np.round(worked_points_coords[:, :2] / pixel_size).astype(int)
        
        # Create mask for valid coordinates within image bounds
        valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < H) & \
                    (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < W)
        
        # Extract valid rows and columns
        rows = pixel_coords[valid_mask, 0]
        cols = pixel_coords[valid_mask, 1]
        
        # Check if we have any valid points
        if len(rows) == 0:
            return None, None, None, None, None
        delta = opt_compression - min_compression
        return (rows, cols), force, opt_compression, delta, bb_points[effectively_worked_mask]
    else:
        return None, None, None, None, None

def solveCircleCollision(points, disk_center, R, max_iters=1):
    """
    Resolve collisions between a disk and a set of points by moving the disk center.
    
    This function handles two scenarios:
    1. For quick, approximate solutions: Uses an iterative repulsion method (active by default)
    2. For precise solutions: Uses convex optimization to find the closest point on the convex hull
    
    The iterative method is currently active as it's significantly faster than optimization.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 2) containing 2D point coordinates that may collide with the disk.
    disk_center : array-like
        Current 2D coordinates [x, y] of the disk center.
    R : float
        Radius of the disk.
    max_iters : int, optional
        Maximum number of iterations for the iterative collision resolution (default: 1).
    
    Returns:
    --------
    numpy.ndarray
        New 2D coordinates [x, y] of the resolved disk center position.
    
    Algorithm (Iterative Method - Currently Active):
    -----------------------------------------------
    1. Identify all points that are inside or touching the disk (obstacles)
    2. Find the closest obstacle point to the disk center
    3. Calculate repulsion vector away from the closest obstacle
    4. Move the disk center along this repulsion vector
    5. Repeat until no collisions or max iterations reached
    
    Algorithm (Optimization Method - Commented Out):
    -----------------------------------------------
    1. Formulate as a convex optimization problem
    2. Find the point on the convex hull of obstacles closest to disk center
    3. Calculate repulsion vector based on this projection
    4. Move disk center to resolve collision
    
    Note:
    -----
    The optimization-based solution was found to be too slow for real-time applications,
    so the faster iterative approach is currently used.
    
    Example:
    --------
    >>> points = np.array([[1, 1], [2, 2], [3, 1], [1.5, 1.5]])  # Obstacle points
    >>> disk_center = np.array([1.8, 1.8])  # Initial disk position
    >>> radius = 0.5
    >>> new_center = solveCircleCollision(points, disk_center, radius)
    >>> print(f"New disk center: {new_center}")
    """
    
    # Iterative Collision Resolution Method (Fast, Currently Active)
    # =============================================================
    
    # Calculate vectors from disk center to all points
    centered = points - disk_center
    # Create mask for points that are inside or touching the disk (collisions)
    obstacles_mask = np.einsum("...k,...k->...", centered, centered) <= R**2#np.linalg.norm(centered, axis=-1) <= R
    
    # Extract the colliding points (obstacles)
    obstacles = centered[obstacles_mask]
    
    # Initialize iteration counter
    k = 0
    
    # Continue resolving collisions while there are obstacles and iterations remain
    while len(obstacles) > 0 and k < max_iters:
        
        # Find the index of the closest obstacle point
        min_idx = np.argmin(np.einsum("...k,...k->...", obstacles, obstacles))#np.linalg.norm(obstacles, axis=-1)
        
        # Calculate vector from closest obstacle to disk center
        v = obstacles[min_idx] - disk_center
        norm = np.linalg.norm(v)
        
        # Handle degenerate case where obstacle is exactly at disk center
        if np.isclose(norm, 0.0):
            # Generate random direction vector
            v = np.random.randn(2)
            norm = np.linalg.norm(v)
        
        # Normalize the repulsion vector
        v = v / norm
        
        # Move disk center away from the obstacle
        # Distance moved = (required separation - current distance + small random offset)
        disk_center -= (R - norm) * v
        
        # Increment iteration counter
        k += 1
        
        # Re-check for collisions with updated disk position
        centered = points - disk_center
        obstacles_mask = np.einsum("...k,...k->...", centered, centered) <= R**2#np.linalg.norm(centered, axis=-1) <= R
        obstacles = centered[obstacles_mask]
    
    # Return the resolved disk center position
    return disk_center

def projectPointsToPlane(points, plane):
    """
    Project 3D points onto a plane.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D point coordinates
    plane : array-like
        Plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
    
    Returns:
    --------
    numpy.ndarray
        Projected points of same shape as input
    """
    plane = plane / np.linalg.norm(plane[:3])
    return points - (np.dot(points, plane[:3]) + plane[3])[:, np.newaxis] * plane[:3]

def projectTo2DPlane(points, origin, plane):
    """
    Project 3D points to a 2D coordinate system on a plane.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D point coordinates
    origin : array-like
        Point [x, y, z] that lies on the plane (origin of 2D coordinate system)
    plane : array-like
        Plane equation coefficients [a, b, c, d]
    
    Returns:
    --------
    tuple
        (coordinates_2d, basis_matrix) where:
        - coordinates_2d: (N, 3) array with z-coordinate being 0 (in plane)
        - basis_matrix: (3, 3) transformation matrix [u, v, normal]
    """
    # Normalize plane and project points
    plane = plane / np.linalg.norm(plane[:3])
    projected_points = points - (np.dot(points, plane[:3]) + plane[3])[:, np.newaxis] * plane[:3]
    
    # Create orthogonal basis vectors on the plane
    u = np.random.randn(3)
    u = u - np.dot(u, plane[:3]) * plane[:3]  # Make u orthogonal to plane normal
    u = u / np.linalg.norm(u)
    v = np.cross(plane[:3], u)  # v orthogonal to both u and plane normal
    
    # Transformation matrix
    B = np.column_stack([u, v, plane[:3]])
    
    # Convert to 2D coordinates
    centered = projected_points - origin
    centered_coordinates = centered @ B
    
    return centered_coordinates, B

def computeDistanceFromPlane(points, plane):
    """
    Compute signed distances from points to a plane.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (N, 3) containing 3D point coordinates
    plane : array-like
        Plane equation coefficients [a, b, c, d]
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (N,) with signed distances
    """
    plane = plane / np.linalg.norm(plane[:3])
    return np.dot(points, plane[:3]) + plane[3]

def is_valid_placement(new_disk_center, disks, min_distance_threshold):
    """
    Return True if new_disk_center is at least min_distance_threshold
    away from all disks.
    """
    if len(disks) == 0:
        return True
    
    disks = np.asarray(disks)  # (N, >=3)
    new_disk_center = np.asarray(new_disk_center)
    
    # pairwise distances
    dists = np.linalg.norm(disks[:, :3] - new_disk_center, axis=1)
    min_dist = dists.min()
    
    return min_dist >= min_distance_threshold

def fit_plane_pca(points, max_population=1000):
    """
    Fit a plane to 3D points using PCA (Principal Component Analysis).
    
    Parameters:
    -----------
    points : array-like
        Array of shape (N, 3) containing 3D point coordinates
    
    Returns:
    --------
    numpy.ndarray
        Plane equation coefficients [a, b, c, d] where ax + by + cz + d = 0
        Normal vector [a, b, c] is normalized and points upward (positive z)
    """
    if len(points) < 3:
        return None
    if max_population is not None and len(points) > max_population:
        points = resample_points(points, max_population)
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    A = pts - centroid
    C = A.T @ A
    _, eigvecs = np.linalg.eigh(C)
    normal = eigvecs[:, 0]  # Eigenvector with smallest eigenvalue
    if normal[2] < 0:
        normal = -normal  # Ensure normal points upward
    normal = normal / np.linalg.norm(normal)
    d = -normal.dot(centroid)
    return np.hstack([normal, d])

def build_graph(disks, max_dist, max_mm_jump=MAX_HEIGHT_CHANGE_MM, max_degree_jump=MAX_NORMAL_CHANGE_DEGREE, max_degree_over_distance = MAX_NORMAL_CHANGE_DEGREE_OVER_DISTANCE):
    """
    Build a directed graph connecting disks based on spatial and orientation constraints.
    
    Creates a graph where disks are nodes and edges represent valid transitions
    between disks based on distance, height difference, and orientation similarity.
    Neighbors are split into 8 angular sectors (45° each) and connected to the
    nearest neighbor in each populated sector.
    
    Parameters:
    -----------
    disks : numpy.ndarray
        Array of shape (N, 6) where each row contains [x, y, z, nx, ny, nz]
        representing disk center coordinates and normal vector components
    max_dist : float
        Maximum Euclidean distance allowed between connected disks
    max_mm_jump : float, optional
        Maximum allowed height difference (in mm) between disks (default: 10.0)
    max_degree_jump : float, optional
        Maximum allowed angle difference (in degrees) between disk orientations (default: 5.0)
    
    Returns:
    --------
    tuple
        (graph, disk2node) where:
        - graph: networkx.DiGraph with nodes representing disks and edges representing valid connections
        - disk2node: dict mapping disk center coordinates (tuple) to node indices
    """
    
    # Initialize directed graph and spatial index tree
    G = nx.DiGraph()
    tree = KDTree(disks[:, :3])  # Index disk centers for fast nearest neighbor search
    disk2node = dict()  # Map disk positions to node indices

    # Add all disks as nodes to the graph
    N = disks.shape[0]
    for i in range(N):
        G.add_node(i)  # Node index i represents disk i
        disk2node[tuple(disks[i, :3])] = i  # Map position tuple to node index

    # Build edges by checking connections for each disk
    for i in range(N):
        # Find more neighbors to ensure we have candidates for all 8 sectors
        dists, idxs = tree.query([disks[i, :3]], k=min(K_NEAREST_NEIGHBORS*2, N), return_distance=True)
        dists = dists[0]  # Extract distances array
        idxs = idxs[0]    # Extract indices array
        
        # Calculate angles for all neighbors relative to disk i
        angles = []
        valid_neighbors = []
        valid_dists = []
        
        # Check each neighbor (skip first which is the disk itself)
        for dist, j in zip(dists[1:], idxs[1:]):
            j = j.item()  # Convert numpy scalar to Python int
            
            # Skip if distance too large
            if dist > max_dist:
                continue
                
            # Skip self-connections
            if i == j:
                continue
            
            # Check orientation compatibility (angle between normals)
            dot_product = np.dot(disks[i][3:6], disks[j][3:6])
            angle_deg = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
            if angle_deg > max_degree_jump:
                continue
            if angle_deg / dist > max_degree_over_distance:
                continue
            
            # Check height compatibility (perpendicular distance between planes)
            height_diff = np.abs(np.dot(disks[j][:3] - disks[i][:3], disks[i][3:6]))
            if height_diff > max_mm_jump:
                continue
            
            # Calculate angle in XY plane from disk i to disk j
            delta = disks[j][:3] - disks[i][:3]
            angle = np.arctan2(delta[1], delta[0])  # Angle in radians
            angle_degrees = np.degrees(angle) % 360  # Convert to 0-360 degrees
            
            angles.append(angle_degrees)
            valid_neighbors.append(j)
            valid_dists.append(dist)
        
        if not valid_neighbors:
            continue
            
        # Split into 8 sectors (0-45°, 45-90°, ..., 315-360°)
        sector_count = SECTOR_COUNT
        sectors = [[] for _ in range(sector_count)]
        sector_distances = [[] for _ in range(sector_count)]
        sector_angle = 360 / sector_count
        for angle, neighbor, dist in zip(angles, valid_neighbors, valid_dists):
            rotated_angle = (angle + sector_angle/2) % 360
            sector_idx = int(rotated_angle // sector_angle) % sector_count  # Which 45° sector (0-7)
            sectors[sector_idx].append(neighbor)
            sector_distances[sector_idx].append(dist)
        
        # Connect to nearest neighbor in each populated sector
        for sector_neighbors, sector_dists in zip(sectors, sector_distances):
            if sector_neighbors:  # If sector has neighbors
                # Find nearest neighbor in this sector
                nearest_idx = np.argmin(sector_dists)
                nearest_neighbor = sector_neighbors[nearest_idx]
                
                # Add bidirectional edges
                if not G.has_edge(i, nearest_neighbor):
                    G.add_edge(i, nearest_neighbor)
                if not G.has_edge(nearest_neighbor, i):
                    G.add_edge(nearest_neighbor, i)
    
    return G, disk2node

def build_opt_model(G, dummy_cost, constrained_edges=None, start_node=None, end_node=None, fast = False):
    """
    Build a mixed-integer linear programming model for finding optimal paths in a graph.
    
    Creates an optimization model that finds a minimum cost path while respecting
    flow conservation and degree constraints. Uses a dummy node to handle open tours.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph with weighted edges
    dummy_cost : float
        Cost of edges connecting to/from dummy node
    constrained_edges : list of tuples, optional
        List of (node1, node2) edges that must be included in the solution
    start_node : int, optional
        Required starting node (connected from dummy)
    end_node : int, optional
        Required ending node (connected to dummy)
    
    Returns:
    --------
    tuple
        (problem, variables) where:
        - problem: PuLP optimization problem
        - variables: dict of decision variables x[i,j] (edge usage)
    
    Model Features:
    ---------------
    - Decision variables: x[i,j] (binary) indicates edge usage
    - Flow variables: f[i,j] (continuous) for MTZ subtour elimination
    - Dummy node connects to all nodes with specified cost
    - Flow conservation constraints ensure valid paths
    - Degree constraints enforce node connectivity
    """
    # Setup
    n_real = len(G.nodes)
    n_real_edges = len(G.edges)
    n = n_real + 1  # +1 for dummy node
    dummy = -1
    
    # Extend graph with dummy node
    extended_G = G.copy()
    extended_G.add_node(dummy)
    for i in extended_G.nodes():
        extended_G.add_edge(i, dummy, weight=dummy_cost)
        extended_G.add_edge(dummy, i, weight=dummy_cost)

    # Create optimization problem
    prob = LpProblem("PathProblem", LpMinimize)

    # Decision variables
    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i, j in extended_G.edges}
    f = {(i, j): LpVariable(f"f_{i}_{j}", lowBound=0, upBound=n - 1, cat=LpContinuous) 
         for i, j in extended_G.edges}
    
    FIX_ALL_SPIRAL_EDGES = fast
    if constrained_edges is not None:
        for spiral_edges in constrained_edges:
            if (spiral_edges[-1][1],spiral_edges[0][0]) in x:
                spiral_edges.append((spiral_edges[-1][1],spiral_edges[0][0]))
            for node1,node2 in spiral_edges:
                if (node1,node2) in x:
                    extended_G.edges[node1,node2]["weight"] /= DISCOUNT_FACTOR
                    x[node1,node2].setInitialValue(1.0)
                    if FIX_ALL_SPIRAL_EDGES:
                        x[node1,node2].fixValue()
            
    # Discourage retracing
    retracing_vars = []
    for node1,node2 in G.edges:
        if G.has_edge(node2,node1):
            retracing_var = LpVariable(f"retracing_penalty_{node1}_{node2}", lowBound=0, upBound=1, cat=LpContinuous)
            x[node1,node2] + x[node2,node1] <= 1 + retracing_var
            retracing_vars.append(retracing_var)

    # Start/end node constraints
    if start_node is not None:
        x[dummy, start_node].setInitialValue(1.0)
        x[dummy, start_node].fixValue()
    if end_node is not None:
        x[end_node, dummy].setInitialValue(1.0)
        x[end_node, dummy].fixValue()
    
    # Objective function: minimize total edge cost
    euclidean_cost = lpSum(extended_G.edges[i, j]["weight"] * x[i, j] for i, j in extended_G.edges)/len(extended_G.edges)
    if retracing_vars:
        retracing_cost = lpSum(var/BACK_EDGE_COST for var in retracing_vars)/len(retracing_vars)
    else:
        retracing_cost = 0
    prob += euclidean_cost + retracing_cost

    # Flow conservation constraints
    for i in extended_G.nodes:
        out_f = lpSum(f[i, j] for j in extended_G.nodes if (i, j) in f)
        in_f = lpSum(f[j, i] for j in extended_G.nodes if (j, i) in f)
        if i == dummy:
            prob += (out_f - in_f == n - 1), f"flow_root"
        else:
            prob += (out_f - in_f == -1), f"flow_node_{i}"

    # Capacity constraints: flow only on used edges
    for i, j in extended_G.edges:
        prob += f[i, j] <= (n - 1) * x[i, j], f"cap_{i}_{j}"

    # Degree constraints
    for i in extended_G.nodes:
        in_deg = lpSum(x[j, i] for j in extended_G.nodes if (j, i) in x)
        out_deg = lpSum(x[i, j] for j in extended_G.nodes if (i, j) in x)
        if i == dummy:
            prob += in_deg == 1, "dummy_in"
            prob += out_deg == 1, "dummy_out"
        else:
            # prob += in_deg >= 1, f"in_deg_{i}"
            prob += in_deg == out_deg, f"deg_balance_{i}"

    return prob, x

def build_opt_model(G, dummy_cost, constrained_edges=None, start_node=None, end_node=None, fast = False):
    """
    Build a mixed-integer linear programming model for finding optimal paths in a graph.
    
    Creates an optimization model that finds a minimum cost path while respecting
    flow conservation and degree constraints. Uses a dummy node to handle open tours.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph with weighted edges
    dummy_cost : float
        Cost of edges connecting to/from dummy node
    constrained_edges : list of tuples, optional
        List of (node1, node2) edges that must be included in the solution
    start_node : int, optional
        Required starting node (connected from dummy)
    end_node : int, optional
        Required ending node (connected to dummy)
    
    Returns:
    --------
    tuple
        (problem, variables) where:
        - problem: PuLP optimization problem
        - variables: dict of decision variables x[i,j] (edge usage)
    
    Model Features:
    ---------------
    - Decision variables: x[i,j] (binary) indicates edge usage
    - Flow variables: f[i,j] (continuous) for MTZ subtour elimination
    - Dummy node connects to all nodes with specified cost
    - Flow conservation constraints ensure valid paths
    - Degree constraints enforce node connectivity
    """
    # Setup
    n_real = len(G.nodes)
    n_real_edges = len(G.edges)
    n = n_real + 1  # +1 for dummy node
    dummy = -1
    
    # Extend graph with dummy node
    extended_G = G.copy().to_directed()
    extended_G.add_node(dummy)
    for i in extended_G.nodes():
        extended_G.add_edge(i, dummy, weight=dummy_cost)
        extended_G.add_edge(dummy, i, weight=dummy_cost)

    # Create optimization problem
    prob = LpProblem("Problem", LpMinimize)

    # Decision variables
    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary) for i, j in extended_G.edges}
    f = {(i, j): LpVariable(f"f_{i}_{j}", lowBound=0, cat=LpContinuous) 
         for i, j in extended_G.edges}
    
    FIX_ALL_SPIRAL_EDGES = fast
    if constrained_edges is not None:
        for spiral_edges in constrained_edges:
            if (spiral_edges[-1][1],spiral_edges[0][0]) in x:
                spiral_edges.append((spiral_edges[-1][1],spiral_edges[0][0]))
            for node1,node2 in spiral_edges:
                if (node1,node2) in x:
                    extended_G.edges[node1,node2]["weight"] /= DISCOUNT_FACTOR
                    x[node1,node2].setInitialValue(1.0)
                    if FIX_ALL_SPIRAL_EDGES:
                        x[node1,node2].fixValue()
            
    # Discourage retracing
    # retracing_vars = []
    # for node1,node2 in G.edges:
    #     if G.has_edge(node2,node1):
    #         retracing_var = LpVariable(f"retracing_penalty_{node1}_{node2}", lowBound=0, upBound=1, cat=LpContinuous)
    #         x[node1,node2] + x[node2,node1] <= 1 + retracing_var
    #         retracing_vars.append(retracing_var)

    # Start/end node constraints
    if start_node is not None:
        x[dummy, start_node].setInitialValue(1.0)
        x[dummy, start_node].fixValue()
    if end_node is not None:
        x[end_node, dummy].setInitialValue(1.0)
        x[end_node, dummy].fixValue()
    
    # Objective function: minimize total edge cost
    euclidean_cost = lpSum(extended_G.edges[i, j]["weight"] * x[i, j] for i, j in extended_G.edges)/len(extended_G.edges)
    # if retracing_vars:
    #     retracing_cost = lpSum(var/BACK_EDGE_COST for var in retracing_vars)/len(retracing_vars)
    # else:
    #     retracing_cost = 0
    prob += euclidean_cost + 0#retracing_cost

    # Flow conservation constraints
    for i in extended_G.nodes:
        if i != dummy:
            out_f = lpSum(f[i, j] for j in extended_G.nodes if (i, j) in f)
            in_f = lpSum(f[j, i] for j in extended_G.nodes if (j, i) in f if j != dummy)
            prob += (out_f - in_f == 1), f"flow_node_{i}"

    # Capacity constraints
    for i, j in extended_G.edges:
        if i != dummy:
            prob += f[i, j] <= (n - 1) * x[i, j], f"cap_{i}_{j}"

    # Degree constraints 
    for i in extended_G.nodes:
        in_deg = lpSum(x[j, i] for j in extended_G.nodes if (j, i) in x)
        out_deg = lpSum(x[i, j] for j in extended_G.nodes if (i, j) in x)
        prob += in_deg == out_deg, f"deg_balance_{i}"
        if i == dummy:
            prob += out_deg == 1, "dummy_out"

    return prob, x

def reconstruct_edge_walk(edges, start):
    """
    Reconstruct an Eulerian path from a list of edges using hierarchical DFS.
    
    Performs a depth-first search to find a valid edge walk (path that uses each edge once),
    prioritizing cycles before returning to parent nodes to ensure proper traversal order.
    
    Parameters:
    -----------
    edges : list of tuples
        List of directed edges [(u, v), ...] representing the graph
    start : int or hashable
        Starting node for the edge walk
    
    Returns:
    --------
    list of tuples
        Ordered list of edges forming a valid walk, or empty list if no valid walk exists
    
    Algorithm:
    ----------
    1. Build adjacency list and track in/out degrees
    2. Use DFS with post-order traversal (visit children before adding edge to path)
    3. Remove edges as they're traversed to avoid revisiting
    4. Reverse final path since edges are added during backtracking
    
    Note:
    -----
    - Assumes the input edges form a valid graph for Eulerian path
    - Uses deque for efficient edge removal
    - Returns edges in traversal order from start to end
    """
    from collections import defaultdict, deque
    
    # Build directed adjacency list and degree counts
    adj = defaultdict(deque)
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)

    for u, v in edges:
        adj[u].append(v)
        out_deg[u] += 1
        in_deg[v] += 1

    path_edges = []  # Stores final edge walk in reverse order

    def dfs(u):
        # Traverse all outgoing edges from node u
        while adj[u]:
            v = adj[u].pop()  # Remove edge (u, v)
            dfs(v)            # Recursively explore from v first
            path_edges.append((u, v))  # Add edge during backtracking

    dfs(start)
    path_edges.reverse()  # Correct the order (was built in reverse)

    return path_edges

def reconstruct_edge_walk(edges, start):
    """
    Iterative version using Hierholzer's algorithm approach.
    """
    from collections import defaultdict, deque
    
    # Build adjacency list
    adj = defaultdict(deque)
    for u, v in edges:
        adj[u].append(v)
    
    path_edges = []
    stack = [start]
    
    while stack:
        u = stack[-1]
        
        if adj[u]:
            # Still has outgoing edges
            v = adj[u].pop()
            stack.append(v)
        else:
            # No more outgoing edges from u
            stack.pop()
            if stack:
                # The edge we just completed is from previous node to u
                prev_node = stack[-1]
                path_edges.append((prev_node, u))
    
    return path_edges[::-1]

def intersect(A, B, C, D, simple = True):
    """
    Check if line segments AB and CD intersect.
    """
    BA = B - A
    DC = D - C
    CA = C - A
    a,b = BA[0],-DC[0]
    c,d = BA[1],-DC[1]
    if simple:
        if np.isclose(a * d - c * b, 0.0):
            return False
        else:
            alpha = CA[0]
            beta = CA[1]
            if (
                (1e-12 < ( alpha * d - beta * b)/(a * d - c * b) < 1.0-1e-12) and 
                (1e-12 < (-alpha * c + beta * a)/(a * d - c * b) < 1.0-1e-12)
            ): # Strict inequality since I always have insersection at the endpoints
                return True
            else:
                return False
    else:
        if np.all(np.isclose(BA, 0.0)) and np.all(np.isclose(DC, 0.0)): # point to point
            return np.all(np.isclose(A,C))
        else:
            if np.all(np.isclose(BA, 0.0)): # point to segment
                if not np.isclose(b, 0.0):
                    t = CA[0] / b
                    if t < 0.0 or t > 1.0:
                        return False
                    return np.isclose(d * t, CA[1])
                else:
                    t = CA[1] / d
                return np.isclose(CA[0], 0.0) and 0 <= t <= 1.0
            elif np.all(np.isclose(DC, 0.0)): #point to segment
                if not np.isclose(a, 0.0):
                    t = CA[0] / a
                    if t < 0.0 or t > 1.0:
                        return False
                    return np.isclose(c * t, CA[1])
                else:
                    t = CA[1] / c
                return np.isclose(CA[0], 0.0) and 0 <= t <= 1.0
            else:
                if np.isclose(a * d - c * b, 0.0): # collinear
                    #Check that they lie on the same line
                    if not np.isclose(a, 0.0): 
                        t = CA[0] / a
                        if not np.isclose(c * t, CA[1]):
                            return False
                    else:
                        t = CA[1] / c
                        if not np.isclose(CA[0], 0.0):
                            return False
                    # C + t DC = A + k BA = p, 0<=t<=1
                    # DC = g BA
                    # CA = k BA - t g BA
                    # CA = z BA, z = k - t g => k = z + t g
                    # For 0<=t<=1, they intersect if at least one k is in [0.0,1.0]
                    if not np.isclose(a,0.0):
                        z = CA[0] / a
                        g = DC[0] / a
                    else:
                        z = CA[1] / c
                        g = DC[1] / c
                    if g>0:
                        if z < 1.0:
                            if z <= 0.0:
                                return 0.0 < z + g <= 1.0
                            else:
                                return True
                        else:
                            return False
                    else:
                        if z > 0.0:
                            if z >= 1.0:
                                return 0.0 < z + g <= 1.0 # Say that backtracking edges don't intersect
                            else:
                                return True
                        else:
                            return False
                else:
                    alpha = CA[0]
                    beta = CA[1]
                    if (
                        (1e-12 < ( alpha * d - beta * b)/(a * d - c * b) < 1.0-1e-12) and 
                        (1e-12 < (-alpha * c + beta * a)/(a * d - c * b) < 1.0-1e-12)
                    ): # Strict inequality since I always have insersection at the endpoints
                        return True
                    else:
                        return False

def intersect_past_edges(A,B, past_segments, simple=True):
    # print()
    for segment in past_segments:
        intersection = intersect(A,B,segment[0],segment[1], simple=simple)
        # print(intersection)
        if intersection:
            return True
    return False

# @timer
def findBoundary(G: nx.DiGraph, points_2d):
    """
    Find boundary nodes of connected components in a 2D directed graph.
    
    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph with 2D coordinates stored in points_2d.
    points_2d : np.ndarray
        (N, 2) array of node positions.
    
    Returns
    -------
    list
        List of node indices forming the boundary for each component.
    """
    all_boundary_nodes = []

    for cc in nx.strongly_connected_components(G):
        cc = G.subgraph(cc).copy()
        nodes = list(cc.nodes)

        if len(nodes) == 1:
            all_boundary_nodes.append([nodes[0]])
            continue

        # Step 1: Find starting point (lowest, then leftmost)
        x0 = min(nodes, key=lambda n: (points_2d[n][1], points_2d[n][0]))

        x_axis = [1.0, 0.0]
        y_axis = [0.0, 1.0]

        # Step 4: Walk along boundary
        boundary_edges = []
        past_segments = []
        current, prev = x0, x0
        max_iterations = int(1e3)
        k = 0

        while k < max_iterations:
            # Candidate edges excluding backtracking and self-intersections
            edges_no_prev = [e for e in cc.edges(current) if e[1] != prev and not intersect_past_edges(points_2d[current], points_2d[e[1]], past_segments)]
            
            if len(edges_no_prev) == 0:
                best_nb = prev
                best_angle = np.pi
                x_axis = -x_axis
                y_axis = -y_axis
            else:
                vecs = np.array([points_2d[nb] - points_2d[current] for (_, nb) in edges_no_prev])
                if len(vecs) == 0:
                    break
                vecs /= np.linalg.norm(vecs, axis=1)[:, None]
                dot_x = np.dot(vecs, x_axis)
                dot_y = np.dot(vecs, y_axis)
                thetas = np.arctan2(dot_y, dot_x)
                best_idx = np.argmin(thetas)
                best_nb = edges_no_prev[best_idx][1]
                best_angle = thetas[best_idx]
                # Rotate local frame
                R = np.array([[np.cos(best_angle), -np.sin(best_angle)],
                            [np.sin(best_angle),  np.cos(best_angle)]])
                x_axis = vecs[best_idx]#R @ x_axis
                y_axis = R @ y_axis
                y_axis = y_axis - np.dot(y_axis,x_axis)*x_axis
                y_axis /= np.linalg.norm(y_axis)

            if best_nb == x0:
                break

            

            # Update
            boundary_edges.append((current, best_nb))
            past_segments.append((points_2d[current], points_2d[best_nb]))
            prev, current = current, best_nb
            k += 1
        temp = [boundary_edges[0][0]]
        temp.extend([e[1] for e in boundary_edges])
        all_boundary_nodes.append(temp)

    return all_boundary_nodes

def assign_weights(cc, disks, min_boundary_nodes = MIN_NODES_FOR_BOUNDARY):
    """
    Assign weights to graph edges based on geometric distance and layer structure.
    
    Computes edge weights as Euclidean distances between disk centers, then
    iteratively identifies boundary layers and reduces weights for intra-layer edges
    to encourage layer-wise connectivity in optimization.
    
    Parameters:
    -----------
    cc : networkx.Graph
        Connected component graph with node indices
    disks : numpy.ndarray
        Array of disk data [x, y, z, nx, ny, nz] for each node
    
    Returns:
    --------
    tuple
        (weighted_graph, constrained_edges) where:
        - weighted_graph: input graph with updated edge weights
        - constrained_edges: list of boundary edge pairs to enforce connectivity
    
    Process:
    --------
    1. Initialize edge weights as 3D Euclidean distances
    2. Iteratively remove boundary layers and identify boundary nodes
    3. Reduce weights for edges connecting nodes in same layer
    4. Collect boundary edges as constrained connections
    """
    temp_G = cc.copy()
    constrained_edges = []
    
    # Set initial weights as 3D distances + normal change
    length_costs = []
    normal_costs = []
    for start, end in cc.edges:
        direction = disks[end, :3] - disks[start, :3]
        length_costs.append(np.linalg.norm(direction))
        normal_change = np.arccos(np.clip(np.dot(disks[end, 3:6],disks[start, 3:6]), -1.0, 1.0))/np.pi
        normal_costs.append(normal_change)
    mean_length_cost = np.mean(length_costs)
    mean_normal_cost = np.mean(normal_costs)
    for start, end in cc.edges:
        direction = disks[end, :3] - disks[start, :3]
        normal_change = np.arccos(np.clip(np.dot(disks[end, 3:6],disks[start, 3:6]), -1.0, 1.0))/np.pi
        cc.edges[start, end]["weight"] = (1-NORMAL_CHANGE_LAMBDA) * np.linalg.norm(direction)/mean_length_cost + NORMAL_CHANGE_LAMBDA * normal_change / mean_normal_cost
    
    # Iteratively process layers
    while len(temp_G.nodes) > 0:
        # Find boundary nodes in current layer
        boundaries = findBoundary(temp_G, disks[:, :2])
        
        # Boundary graph
        boundary_graph = nx.DiGraph()
        undirected_boundary_graph = nx.Graph()
        for boundary_nodes in boundaries:
            # Close the loop if it's a cycle
            edges = [(boundary_nodes[i], boundary_nodes[i+1]) 
                    for i in range(len(boundary_nodes) - 1)]
            if len(boundary_nodes) > 2 and (boundary_nodes[-1], boundary_nodes[0]) in temp_G.edges:
                edges.append((boundary_nodes[-1], boundary_nodes[0]))


            boundary_graph.add_nodes_from(boundary_nodes)
            boundary_graph.add_edges_from(edges)
            undirected_boundary_graph.add_nodes_from(boundary_nodes)
            undirected_boundary_graph.add_edges_from(edges)
                
        # Keep removing until no more loose nodes found
        changed = True
        while changed:
            changed = False
            nodes_to_remove = []
            for node in list(undirected_boundary_graph.nodes()):
                node_degree = undirected_boundary_graph.degree(node)
                # Remove nodes with insufficient connections
                if node_degree <= 1:
                    nodes_to_remove.append(node)
                    changed = True
            
            undirected_boundary_graph.remove_nodes_from(nodes_to_remove)
            boundary_graph.remove_nodes_from(nodes_to_remove)
                
        connected_components = nx.weakly_connected_components(boundary_graph)
        for component in connected_components:
            component = boundary_graph.subgraph(component)
            if len(component.nodes) > min_boundary_nodes:
                constrained_edges.append(list(component.edges))
        # Remove current layer and continue
        temp_G.remove_nodes_from([node for boundary_nodes in boundaries for node in boundary_nodes])
        
    
    return cc, constrained_edges
# @timer
def fast_voxel_grid_sampling(point_cloud, min_distance):
    # Different voxel size for each dimension
    voxel_coords = np.floor(point_cloud / np.array(min_distance)).astype(int)
    
    # Create unique keys
    keys = (voxel_coords[:, 0] * 1000000 + 
            voxel_coords[:, 1] * 1000 + 
            voxel_coords[:, 2])
    
    # Sample one point per voxel
    _, unique_indices = np.unique(keys, return_index=True)
    
    return point_cloud[unique_indices]

def plot_disks_and_point_cloud(point_cloud, disks, interactive=True, number=0):
    disks = np.array(disks)
    plotter = pv.Plotter()
    z_coords = point_cloud[:, 2]
    z_min, z_max = np.percentile(z_coords, [5, 95])
    z_coords = z_max - point_cloud[:, 2] + z_min
    plotter.add_points(point_cloud, point_size=2, render_points_as_spheres=False, opacity=0.3, scalars=z_coords, cmap="viridis",clim=[z_min, z_max])
    polydata = add_disks_as_points_and_lines(
        plotter, disks[:,:3], disks[:,3:6], 0, disks[:,7], 
        c_res=24, color="black", opacity=0.5
    )
    plotter.view_yx()
    plotter.camera.view_up = [0, -1, 0]
    plotter.camera.elevation = -30
    plotter.zoom_camera(1.5)
    plotter.add_axes()
    plotter.save_graphic(f"out/{number}.pdf")
    plotter.show(interactive=interactive, auto_close=False)

def debug_collision_check(i, ball_points, plane, disk_center, perp_distances):
    collision_check_summary = lambda i, collision_detected, outliers_count: f"Disk {i}: Collision={'YES' if collision_detected else 'NO'}, Outliers={outliers_count}"

    # 2. Volume calculation one-liner
    collision_volume = lambda perp_distances, threshold: np.sum(-perp_distances[perp_distances < -threshold]) - threshold * np.sum(perp_distances < -threshold) if np.sum(perp_distances < -threshold) > 0 else 0

    # 3. Quick outlier statistics
    outlier_stats = lambda perp_distances, threshold: f"Outliers: {np.sum(perp_distances < -threshold)}/{len(perp_distances)} ({100*np.sum(perp_distances < -threshold)/len(perp_distances):.1f}%)"

    # 4. Distance range checker
    distance_range = lambda perp_distances: f"Dist range: [{np.min(perp_distances):.2f}, {np.max(perp_distances):.2f}]"

    """Debug collision checking with detailed output"""
    outliers_mask = perp_distances < -COLLISION_THRESHOLD_MM
    outliers_count = np.sum(outliers_mask)
    collision_vol = collision_volume(perp_distances, COLLISION_THRESHOLD_MM)
    collision_detected = collision_vol > PENETRATION_VOLUME_THRESHOLD
    
    print(f"\n=== Disk {i} Collision Check ===")
    print(f"Center: {disk_center}")
    print(f"Points: {len(ball_points)}")
    print(f"Plane valid: {plane is not None and np.linalg.norm(plane[:3]) > 0.1}")
    print(f"{outlier_stats(perp_distances, COLLISION_THRESHOLD_MM)}")
    print(f"{distance_range(perp_distances)}")
    print(f"Volume: {collision_vol:.3f} (threshold: {PENETRATION_VOLUME_THRESHOLD})")
    print(f"Collision: {'DETECTED' if collision_detected else 'NOT DETECTED'}")
    
    return collision_detected, outliers_mask


def visualize_collision_points(pc,disk_center, ball_points, outliers_mask, plane, title="Collision Debug"):
    """Visualize collision points using PyVista"""
    try:
        import pyvista as pv
        
        plotter = pv.Plotter()
        
        # Add all points (blue)
        all_points_cloud = pv.PolyData(pc)
        plotter.add_mesh(all_points_cloud, color='blue', point_size=8, render_points_as_spheres=True, opacity=0.6)
        
        # Add outlier points (red)
        if np.sum(outliers_mask) > 0:
            outlier_points = ball_points[outliers_mask]
            outlier_cloud = pv.PolyData(outlier_points)
            plotter.add_mesh(outlier_cloud, color='red', point_size=12, render_points_as_spheres=True)
        
        # Add disk center (green)
        center_sphere = pv.Sphere(radius=DISK_RADIUS_MM/10, center=disk_center)
        plotter.add_mesh(center_sphere, color='green')
        
        # Add plane visualization
        if plane is not None:
            # Create a disk to represent the plane
            disk_center_point = disk_center.copy()
            disk_center_point[2] = -(plane[0]*disk_center[0] + plane[1]*disk_center[1] + plane[3]) / plane[2] if abs(plane[2]) > 1e-10 else disk_center[2]
            
            # Create disk points
            theta = np.linspace(0, 2*np.pi, 32)
            u = np.array([1, 0, 0]) if abs(plane[2]) < 0.9 else np.array([1, 0, 0])
            v = np.cross(plane[:3], u)
            v = v / np.linalg.norm(v)
            u = np.cross(plane[:3], v)
            u = u / np.linalg.norm(u)
            
            disk_points = []
            for angle in theta:
                point_on_disk = disk_center_point + DISK_RADIUS_MM * (np.cos(angle) * u + np.sin(angle) * v)
                disk_points.append(point_on_disk)
            
            disk_poly = pv.PolyData(np.array(disk_points))
            plotter.add_mesh(disk_poly, color='green', opacity=0.3)
        
        plotter.add_axes()
        plotter.show(title=title, interactive=True)
        
    except ImportError:
        print("PyVista not available for visualization")

if __name__ == "__main__":
    import pyvista as pv
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KDTree
    from shapely.geometry import Polygon
    from shapely import affinity
    
    _,_,z_matrix = read_depthmap(DEPTHMAP_FILENAME)
    z_matrix = z_matrix # (N,M)
    full_surface_point_cloud = depth_to_pointcloud(z_matrix)
    full_point_cloud = full_surface_point_cloud.reshape(-1,3)

    min_y, max_y = np.min(full_point_cloud[:,1]), np.max(full_point_cloud[:,1])
    min_x, max_x = np.min(full_point_cloud[:,0]), np.max(full_point_cloud[:,0])
    work_area_polygon = np.array([
        (min_x,min_y),
        (max_x,min_y),
        (max_x,max_y),
        (min_x,max_y)
    ])
    poly = Polygon(work_area_polygon)
    shrunk_poly = poly.buffer(-DISK_RADIUS_MM)
    if shrunk_poly.is_empty:
        raise Exception("Empty workspace")
    shrunk_coords = np.array(shrunk_poly.exterior.coords)
    work_area_px = (shrunk_coords / pixel_size).astype(np.int32)
    import cv2
    work_area = np.zeros(full_surface_point_cloud.shape[:2], dtype=np.uint8)
    cv2.fillPoly(work_area, [work_area_px[:,::-1]], 255)
    outer_area = ~work_area
    z_matrix[outer_area > 0] = np.nan
    surface_point_cloud = depth_to_pointcloud(z_matrix) # (N,M,3)
    nan_mask = np.isnan(surface_point_cloud[:,:,2])
    point_cloud = surface_point_cloud[~nan_mask].reshape(-1,3) # (n,3)
    tree = KDTree(full_point_cloud)


    r = R_MM
    sampled_points = fast_voxel_grid_sampling(point_cloud, [SAMPLING_DISTANCE,SAMPLING_DISTANCE,SAMPLING_DISTANCE])
    indices_in_radius = tree.query_radius(sampled_points, r=r)
    small_disks = []
    for i,neighbors_idxs in tqdm(enumerate(indices_in_radius),desc="Finding well placed small disks...", ncols = 100, total=len(indices_in_radius)):
        point = sampled_points[i]
        small_ball_points = full_point_cloud[neighbors_idxs]
        if len(neighbors_idxs) < MIN_POINTS:
            continue
        plane = fit_plane_pca(small_ball_points, max_population=None)
        if plane is None:
            continue
        # Check small disk fit
        perp_distances = computeDistanceFromPlane(small_ball_points, plane)
        outliers_mask = perp_distances < -SMALL_DISK_INLIER_THRESHOLD
        if np.any(outliers_mask):
            continue
        disk_center = point - computeDistanceFromPlane(point, plane) * plane[:3]
        small_disks.append(np.hstack([disk_center,plane,r]))

    small_disks = np.array(small_disks)
    # plot_disks_and_point_cloud(full_point_cloud, small_disks,interactive=True)

    r = DISK_RADIUS_MM
    collision_disks = []
    indices_in_radius = tree.query_radius(small_disks[:,:3], r=r)
    no_collision_disks = []
    for i,neighbors_idxs in tqdm(enumerate(indices_in_radius),desc="Solving collisions...", ncols = 100, total=len(indices_in_radius)):
        disk_center = small_disks[i,:3]
        plane = small_disks[i,3:7]
        ball_points = full_point_cloud[neighbors_idxs]#bb_points[points_mask]
        perp_distances = computeDistanceFromPlane(ball_points, plane)
        outliers_mask = perp_distances < -COLLISION_THRESHOLD_MM
        collision_detected = np.sum(-perp_distances[outliers_mask]) - COLLISION_THRESHOLD_MM*perp_distances[outliers_mask].shape[0] > PENETRATION_VOLUME_THRESHOLD
        if collision_detected:
            # Refit plane
            perp_distances = computeDistanceFromPlane(ball_points, plane)
            inliers_mask = np.abs(perp_distances) < COLLISION_THRESHOLD_MM
            plane = fit_plane_ransac_implicit(ball_points[inliers_mask])
            # plane = fit_plane_pca(ball_points[inliers_mask])
            if plane is None:
                continue
            perp_distances = computeDistanceFromPlane(ball_points, plane)
            outliers_mask = perp_distances < -COLLISION_THRESHOLD_MM
            outliers = ball_points[outliers_mask]
            # Solve collision 
            centered_outliers_coordinates, B = projectTo2DPlane(outliers, disk_center, plane)
            center_mm = solveCircleCollision(centered_outliers_coordinates[:, :2], 
                                        np.array([0.0, 0.0]), 
                                        DISK_RADIUS_MM, max_iters=3)
            center_displacement = (B[:, :2] @ center_mm[:, np.newaxis]).squeeze()
            disk_center = disk_center + center_displacement
            if np.linalg.norm(center_displacement) > 0.75*DISK_RADIUS_MM:
                continue
            # Check if new disk position is valid
            if not is_valid_placement(disk_center,collision_disks,R_MM/2):
                continue
            if work_area[*(disk_center[:2]/pixel_size).astype(int)] == 0:
                continue
            collision_disks.append(np.hstack([disk_center,plane,r]))
        else:
            inliers_mask = np.abs(perp_distances) < COLLISION_THRESHOLD_MM
            plane = fit_plane_ransac_implicit(ball_points[inliers_mask])
            # plane = fit_plane_pca(ball_points[inliers_mask])
            if plane is None:
                continue
            
            disk_center = disk_center - computeDistanceFromPlane(disk_center, plane) * plane[:3]
            no_collision_disks.append(np.hstack([disk_center,plane,r]))
    collision_disks = np.array(collision_disks)
    plot_disks_and_point_cloud(full_point_cloud, collision_disks,    interactive=True, number=0)
    # plot_disks_and_point_cloud(full_point_cloud, no_collision_disks, interactive=True)

    r = DISK_RADIUS_MM
    indices_in_radius = tree.query_radius(collision_disks[:,:3], r=r)
    for i,neighbors_idxs in tqdm(enumerate(indices_in_radius),desc="Checking collisions...", ncols = 100, total=len(indices_in_radius)):
        disk_center = collision_disks[i,:3]
        plane = collision_disks[i,3:7]
        x_interval, y_interval, bb_points, points_mask = get_column(full_surface_point_cloud, disk_center, plane[:3], column_height=4*DISK_RADIUS_MM, asymmetric_ratio=3.0, radius=DISK_RADIUS_MM)
        if bb_points is None:
            continue
        column_points = bb_points[points_mask]#full_point_cloud[neighbors_idxs]

        perp_distances = computeDistanceFromPlane(column_points, plane)
        outliers_mask = perp_distances < -COLLISION_THRESHOLD_MM
        inliers_mask = np.abs(perp_distances) < COLLISION_THRESHOLD_MM
        collision_detected = np.sum(-perp_distances[outliers_mask]) - COLLISION_THRESHOLD_MM*perp_distances[outliers_mask].shape[0] > PENETRATION_VOLUME_THRESHOLD
        # Skip if collision is still present
        if not collision_detected and np.count_nonzero(inliers_mask) > 50*50:
            no_collision_disks.append(np.hstack([disk_center,plane,r]))
        
    # if no_collision_disks:
    #     plot_disks_and_point_cloud(full_point_cloud, no_collision_disks)



    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()

    no_collision_disks = np.array(no_collision_disks)[::-1,:]
    delta_compressions = []
    space = np.ones(full_surface_point_cloud.shape[:2],dtype=np.bool)
    collision_free_disks = []
    r = DISK_RADIUS_MM
    indices_in_radius = tree.query_radius(no_collision_disks[:,:3], r=r)
    for i,neighbors_idxs in tqdm(enumerate(indices_in_radius),desc="Filling...", ncols = 100, total=len(indices_in_radius)):
        disk_center = no_collision_disks[i,:3]
        plane = no_collision_disks[i,3:7]
        column_points = full_point_cloud[neighbors_idxs]
        perp_distances = computeDistanceFromPlane(column_points, plane)
        inliers_mask = perp_distances < SMALL_DISK_INLIER_THRESHOLD
        region = column_points[inliers_mask]
        region_idxs = np.round(region[:,:2]/pixel_size).astype(int)
        space_region = space[region_idxs[:,0],region_idxs[:,1]]
        space_region_size = space_region.shape[0]
        not_worked_count = np.count_nonzero(space_region)
        worked_count = space_region_size - not_worked_count
        if worked_count > 0.95 * space_region_size: # More than 95% of the region was already worked
            continue
        # if not is_valid_placement(disk_center,collision_free_disks,SAMPLING_DISTANCE):
        #     continue
        worked_indices,pf,opt_compression, delta_compression,column_points = worked_points(full_surface_point_cloud, disk_center, DISK_RADIUS_MM, plane[:3], force_threshold=None, max_pressure=REFERENCE_PRESSURE)
        # Skip if worked area is too small
        if worked_indices is None:
            continue
        if pf < MIN_ACTUATOR_FORCE:
            continue
        disk_center = disk_center+opt_compression*plane[:3]
        current_working_space = space[worked_indices]
        current_working_space_size = current_working_space.shape[0]
        currently_working_size = np.count_nonzero(current_working_space)
        already_worked = current_working_space_size - currently_working_size
        if not (currently_working_size > MIN_NEW_WORKED_AREA * space_region_size):
            continue
        # Skip if it's working an already worked region
        # if not (already_worked < WORKED_POINTS_OVERLAP_THRESHOLD * space[worked_indices].shape[0]):
        #     continue
        space[worked_indices] = 0
        collision_free_disks.append(np.hstack([disk_center,plane,r,pf]))
        delta_compressions.append(delta_compression)
    print("Average delta compression: ", np.mean(delta_compressions))   
    
    # if collision_free_disks:
    #     plot_disks_and_point_cloud(full_point_cloud, collision_free_disks) 
    plt.figure()
    plt.imshow(space)
    plt.savefig("out/worked.png")
    admissible_disks = np.array(collision_free_disks)
    print(f"Final disks count: {admissible_disks.shape[0]}")
    assert admissible_disks.shape[0] > 0
    # pv.set_jupyter_backend("client")
    if os.path.exists(os.path.join("out", "disks.png")):
        os.remove(os.path.join("out", "disks.png"))
    for filename in os.listdir("out"):
        if filename.startswith("route_"):
            os.remove(os.path.join("out",filename))
    for filename in os.listdir("out"):
        if filename.startswith("animation_"):
            os.remove(os.path.join("out",filename))

    

    if ENABLE_PLOTS:
        plot_disks_and_point_cloud(full_point_cloud, admissible_disks, number=1)
    else:
        plotter = pv.Plotter(off_screen=True)
        z_coords = full_point_cloud[:, 2]
        z_min, z_max = np.percentile(z_coords, [5, 95])
        z_coords = z_max - full_point_cloud[:, 2] + z_min
        plotter.add_points(full_point_cloud, point_size=2, render_points_as_spheres=False, opacity=0.3, scalars=z_coords, cmap="viridis",clim=[z_min, z_max])
        
        polydata = add_disks_as_points_and_lines(
            plotter, admissible_disks[:,:3], admissible_disks[:,3:6], 0, DISK_RADIUS_MM, 
            c_res=24, color="black", opacity=0.5
        )
        plotter.view_yx()
        plotter.camera.view_up = [0, -1, 0]
        plotter.camera.elevation = -30
        plotter.zoom_camera(1.5)
        plotter.screenshot("out/disks.png")

    G, disk2node = build_graph(admissible_disks, CONNECTIVITY_RADIUS)
    ccs = list(nx.strongly_connected_components(G))
    paths = []
    for cc_id,cc in enumerate(ccs):
        if len(cc) > MINIMUM_ROUTE_LENGTH:
            cc = G.subgraph(cc)
            print(f"Working on {len(cc.nodes)} nodes and {len(cc.edges)} edges")
            central_point = np.mean(admissible_disks[cc.nodes,:3], axis=0)
            cc, constrained_edges = assign_weights(cc, admissible_disks)
            nodes = list(cc.nodes)
            if constrained_edges:
                spiral_nodes = set()
                for spiral_edges in constrained_edges:
                    for u,v in spiral_edges:
                        spiral_nodes.add(u)
                        spiral_nodes.add(v)
                spiral_nodes = list(spiral_nodes)
                starting_node = spiral_nodes[np.argmin(np.linalg.norm(admissible_disks[spiral_nodes,:2],axis=-1))]#constrained_edges[0][0][0]
            else:
                starting_node = nodes[np.argmin(np.linalg.norm(admissible_disks[nodes,:2],axis=-1))]#constrained_edges[0][0][0]
            if OPTIMIZE_END_NODE:
                ending_node = None
            else:
                if constrained_edges: 
                    ending_node = constrained_edges[-1][-1][1]
                else:
                    ending_node = None
            if len(cc) > RELAXED_PROBLEM_SIZE_THRESHOLD:
                prob, x_vars = build_opt_model(cc, DUMMY_COST, start_node=starting_node, end_node=ending_node, constrained_edges=constrained_edges, fast = True if len(cc.edges)>2000 else False)
            else:
                prob, x_vars = build_opt_model(cc, DUMMY_COST, start_node=starting_node, end_node=None, constrained_edges=constrained_edges)
            start_time = time.time()
            status = prob.solve(SCIP_PY(msg=0, timeLimit=15.0,gapRel=RELATIVE_GAP_THRESHOLD))
            var_dict = prob.variablesDict()
            if status != LpStatusOptimal:
                print("Optimization failed.")
                if time.time() - start_time < 10.:
                    print("Retrying with relaxed constraints...")
                    prob, x_vars = build_opt_model(cc, DUMMY_COST, start_node=starting_node, end_node=None, constrained_edges=None, fast = False)
                    status = prob.solve(SCIP_PY(msg=False, timeLimit=15.0,gapRel=RELATIVE_GAP_THRESHOLD))
                    if status != LpStatusOptimal:
                        print("Optimization failed again, a path was not found, skipping route...")
                    else:
                        print("Optimization succeeded with related constraints...")
                else:
                    continue
            print(f"Path optimization took {time.time() - start_time} seconds")
            dummy = -1
            path = [(i, j) for (i, j), v in x_vars.items() if v.value() > 0.5]
            path = reconstruct_edge_walk(path, -1)[1:-1]
            starting_node = path[0][0]
            ending_node = path[-1][-1]
            
            node_path = [starting_node]
            for (u,v) in path:
                node_path.append(v)
            paths.append(node_path)
            # Plot
            start_time = time.time()
            if ENABLE_PLOTS:
                plot_and_animate(full_point_cloud, admissible_disks, path, f"out/animation_{cc_id}.mp4", constrained_edges=None)#plot_and_animate(full_point_cloud, admissible_disks, path, f"out/animation_{cc_id}.mp4", constrained_edges=constrained_edges)
            else:
                plotter = pv.Plotter(off_screen=True)
                z_coords = full_point_cloud[:, 2]
                z_min, z_max = np.percentile(z_coords, [5, 95])
                z_coords = z_max - full_point_cloud[:, 2] + z_min
                plotter.add_points(full_point_cloud, point_size=2, render_points_as_spheres=False, opacity=0.3, scalars=z_coords, cmap="viridis",clim=[z_min, z_max])
                plotter.add_points(admissible_disks[starting_node][:3], color="green", point_size=10, render_points_as_spheres=False)
                plotter.add_points(admissible_disks[ending_node][:3], color="red", point_size=10, render_points_as_spheres=False)
                _disks = []
                for node in cc:
                    disk = admissible_disks[node]
                    _disks.append(disk)
                _disks = np.array(_disks)
                polydata = add_disks_as_points_and_lines(
                    plotter, _disks[:,:3], _disks[:,3:6], R_MM, DISK_RADIUS_MM, 
                    c_res=24, color="black", opacity=0.5
                )
                
                # plotter.add_lines(np.array([(disks[i][:3], disks[j][:3]) for path in constrained_edges for i ,j in path]).reshape(-1,3), color="purple", width=3)
                plotter.add_lines(np.array([(admissible_disks[i][:3]-[0,0,10], admissible_disks[j][:3]-[0,0,10]) for i ,j in path]).reshape(-1,3), color="red", width=3)
                plotter.view_yx()
                plotter.camera.view_up = [0, -1, 0]
                plotter.zoom_camera(1.5)
                plotter.screenshot(f"out/route_{cc_id}.png")
            print(f"Plotting took {time.time() - start_time} seconds")
            

    try:
        plotter.close()
        del plotter
    except Exception:
        pass

    
    import json

    # Build the routes structure
    if os.path.exists(ROUTES_JSON_OUTPUT_FILEPATH):
        if os.path.exists(ROUTES_JSON_OUTPUT_FILEPATH + ".auto_bkp"):
            os.remove(ROUTES_JSON_OUTPUT_FILEPATH + ".auto_bkp")
        os.rename(ROUTES_JSON_OUTPUT_FILEPATH, ROUTES_JSON_OUTPUT_FILEPATH + ".auto_bkp")
    routes = []
    
    for node_path in paths:
        route = []
        for node in node_path:
            # res = worked_points(point_cloud, coords[:3], DISK_RADIUS_MM, coords[3:6], pixel_size=1.125, force_threshold=1500.0, max_pressure = 2.5)
            # print(res)
            # _, force, pressure = res
            coords = admissible_disks[node].copy()
            # x y z nx ny nz d 
            x,y,z,normal_x,normal_y,normal_z,d,r,pf = coords
            if np.isclose(normal_z,0.,atol=1e-4):
                continue
            route.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "a": float(normal_x/normal_z),
                "b": float(normal_y/normal_z),
                "c": float(d),
                "pf": float(pf)
            })
        routes.append(route)

    # Create the final dictionary
    output_data = {"routes": routes}

    # Save to file
    with open(ROUTES_JSON_OUTPUT_FILEPATH, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Script executed in {time.time() - script_start_time} seconds")

