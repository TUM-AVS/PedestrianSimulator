__author__ = "Korbinian Moller"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib as mpl
from wale_net_lite.visualization import confidence_ellipse


def monte_carlo_collision_probability(ego_pos, ego_orientation, ego_dimensions, obstacle_mean, obstacle_cov,
                                      num_samples=10000, plot=False, zorder=100):
    """
    Computes the collision probability using Monte Carlo sampling (without considering obstacle size).
    Treats the obstacle as a point with positional uncertainty.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle.
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        ego_dimensions (tuple): (length, width) dimensions of the ego vehicle.
        obstacle_mean (np.ndarray): [x, y] expected position of the pedestrian's center.
        obstacle_cov (np.ndarray): 2x2 covariance matrix of the pedestrian's position.
        num_samples (int): Number of Monte Carlo samples to generate.
        plot (bool): Whether to plot the pedestrian's position and samples.
        zorder (int): Z-order for plotting the pedestrian's position and samples.

    Returns:
        float: Estimated collision probability using Monte Carlo sampling.
    """
    # Unpack dimensions
    ego_length, ego_width = ego_dimensions

    # Rotation matrix to align the ego vehicle's coordinate frame with the global frame
    cos_theta = np.cos(-ego_orientation)  # Negative sign for inverse rotation
    sin_theta = np.sin(-ego_orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])

    # Define the ego vehicle's bounding box in its own coordinate frame (no obstacle size)
    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    lower_bound = np.array([-half_length, -half_width])
    upper_bound = np.array([half_length, half_width])

    # Generate samples from the pedestrian's multivariate normal distribution
    samples = np.random.multivariate_normal(mean=obstacle_mean, cov=obstacle_cov, size=num_samples)

    # Transform each sample to the ego vehicle's coordinate frame
    relative_samples = samples - ego_pos
    transformed_samples = (rotation_matrix @ relative_samples.T).T

    # Count how many samples fall inside the ego vehicle's bounding box
    inside_box = np.all((transformed_samples >= lower_bound) & (transformed_samples <= upper_bound), axis=1)
    num_inside_box = np.sum(inside_box)

    world_samples_inside = (rotation_matrix.T @ transformed_samples[inside_box].T).T + ego_pos

    # Estimate collision probability as the proportion of samples inside the box
    collision_probability = num_inside_box / num_samples

    if plot:
        plot_ego_vehicle(ego_pos, ego_orientation, ego_dimensions, zorder=zorder-1)
        plt.plot(obstacle_mean[0], obstacle_mean[1], 'ro', zorder=zorder)
        confidence_ellipse(mu=obstacle_mean, cov=obstacle_cov, ax=plt.gca(), n_std=3, facecolor='blue', zorder=zorder)
        plt.plot(samples[:, 0], samples[:, 1], 'ko', zorder=zorder)
        plt.plot(world_samples_inside[:, 0], world_samples_inside[:, 1], 'go', zorder=zorder)

    return collision_probability


def compute_collision_probability(ego_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov):
    """
    Computes the collision probability between the ego vehicle and a pedestrian at a given time step.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle.
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        ego_dimensions (tuple): (length, width) dimensions of the ego vehicle.
        pedestrian_mean (np.ndarray): [x, y] expected position of the pedestrian.
        pedestrian_cov (np.ndarray): 2x2 covariance matrix of the pedestrian's position.

    Returns:
        float: Collision probability between 0 and 1.
    """
    # Unpack the ego vehicle dimensions
    ego_length, ego_width = ego_dimensions

    # Rotation matrix to align the ego vehicle's coordinate frame with the global frame
    cos_theta = np.cos(-ego_orientation)  # Negative sign for inverse rotation
    sin_theta = np.sin(-ego_orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])

    # Transform the pedestrian's mean position into the ego vehicle's coordinate frame
    relative_mean = pedestrian_mean - ego_pos
    transformed_mean = rotation_matrix @ relative_mean

    # Transform the pedestrian's covariance matrix into the ego vehicle's coordinate frame
    transformed_cov = rotation_matrix @ pedestrian_cov @ rotation_matrix.T

    # Define the collision zone (ego vehicle's bounding box in its own coordinate frame)
    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    lower = np.array([-half_length, -half_width])
    upper = np.array([half_length, half_width])

    # Compute the cumulative distribution function (CDF) at the four corners of the rectangle
    cdf_upper_upper = multivariate_normal.cdf(upper, mean=transformed_mean, cov=transformed_cov)
    cdf_lower_upper = multivariate_normal.cdf([lower[0], upper[1]], mean=transformed_mean, cov=transformed_cov)
    cdf_upper_lower = multivariate_normal.cdf([upper[0], lower[1]], mean=transformed_mean, cov=transformed_cov)
    cdf_lower_lower = multivariate_normal.cdf(lower, mean=transformed_mean, cov=transformed_cov)

    # Apply the inclusion-exclusion principle to compute the probability over the rectangle
    collision_probability = (
        cdf_upper_upper
        - cdf_lower_upper
        - cdf_upper_lower
        + cdf_lower_lower
    )

    # Ensure the probability is within [0, 1]
    if collision_probability < 0.0 or collision_probability > 1.0:
        if collision_probability > 1.0:
            print(f"Warning: Collision probability {collision_probability} is out of bounds. Correcting it.")
        # Clip the probability to the valid range
        collision_probability = max(min(collision_probability, 1.0), 0.0)

    return np.round(collision_probability, 4)


def get_collision_probability(traj, predictions, vehicle_params, safety_margin=0.0, debug=False):
    """
    Calculates the collision probabilities of a trajectory with predicted pedestrian positions.

    Args:
        traj (FrenetTrajectory): The considered trajectory of the ego vehicle.
        predictions (dict): Predictions of visible pedestrians, including position and covariance.
        vehicle_params (VehicleParameters): Parameters of the ego vehicle (length, width, etc.).
        safety_margin (float): Additional safety margin to consider around the vehicle dimensions.

    Returns:
        dict: Collision probability per time step for each visible pedestrian.
    """
    collision_prob_dict = {}
    ego_dimensions = (vehicle_params.length + safety_margin, vehicle_params.width + safety_margin)
    ego_positions = np.stack((traj.cartesian.x, traj.cartesian.y), axis=-1)
    ego_orientations = traj.cartesian.theta

    for pedestrian_id, prediction in predictions.items():  #ToDo Add distance check for faster compuation
        probs = []
        mean_list = prediction['pos_list']
        cov_list = prediction['cov_list']
        min_len = min(len(traj.cartesian.x), len(mean_list))

        for i in range(min_len):
            # store values in temporary variables
            ego_orientation = ego_orientations[i]
            ego_pos = ego_positions[i]
            ego_center_pos = shift_to_vehicle_center(ego_pos, ego_orientation, vehicle_params.wb_rear_axle)
            pedestrian_mean = np.array(mean_list[i][:2])
            pedestrian_cov = np.array(cov_list[i][:2, :2])

            # Ensure that the covariance matrix is positive semi-definite
            pedestrian_cov = ensure_positive_semi_definite(pedestrian_cov)

            # Compute the collision probability using the CDF method (analytical)
            prob = compute_collision_probability(
                ego_center_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov
            )

            if debug:
                # Validation of the collision probability using Monte Carlo sampling
                monte_carlo_prob = monte_carlo_collision_probability(
                    ego_center_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov, num_samples=10000, plot=False
                )

                if not np.isclose(prob, monte_carlo_prob, atol=0.01):
                    print(f"Collision Probability: {prob}")
                    print(f"Monte Carlo Collision Probability: {monte_carlo_prob}")

            probs.append(prob)

        collision_prob_dict[pedestrian_id] = np.array(probs)

    return collision_prob_dict


### Helper Functions ###

def ensure_positive_semi_definite(cov_matrix):
    """
    Ensure the covariance matrix is positive semi-definite.
    If it's not, adjust by adding a small value to the diagonal.

    Args:
        cov_matrix (np.ndarray): Covariance matrix to check and adjust.

    Returns:
        np.ndarray: Adjusted covariance matrix that is positive semi-definite.
    """
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # If decomposition fails, add small value to diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    return cov_matrix


def shift_to_vehicle_center(ego_pos, ego_orientation, wheelbase_rear_axle):
    """
    Shifts the reference point of the ego vehicle from the rear axle to the center of the vehicle.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle's rear axle (from trajectory).
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        wheelbase_rear_axle (float): Distance from the rear axle to the center of the vehicle (wheelbase length).

    Returns:
        np.ndarray: The [x, y] position of the ego vehicle's center.
    """
    # Compute the forward shift along the vehicle's orientation direction (i.e., x-axis in the ego vehicle's local frame)
    dx = wheelbase_rear_axle * np.cos(ego_orientation)
    dy = wheelbase_rear_axle * np.sin(ego_orientation)

    # Shift the ego vehicle's position from rear axle to center
    center_pos = np.array([ego_pos[0] + dx, ego_pos[1] + dy])

    return center_pos


def plot_ego_vehicle(ego_pos, ego_orientation, ego_dimensions, color='blue', label='Ego Vehicle', zorder=100, **kwargs):
    """
    Plots the ego vehicle as a rectangle given its position, orientation, and dimensions.

    Args:
        ego_pos (np.ndarray): [x, y] position of the ego vehicle.
        ego_orientation (float): Orientation angle (theta) of the ego vehicle in radians.
        ego_dimensions (tuple): (length, width) dimensions of the ego vehicle.
        color (str): Color for plotting the ego vehicle.
        label (str): Label for the ego vehicle in the plot legend.
    """
    plt.axis('equal')

    # Unpack the ego vehicle dimensions
    ego_length, ego_width = ego_dimensions

    # Define the corners of the vehicle in its local coordinate frame
    half_length = ego_length / 2.0
    half_width = ego_width / 2.0
    corners = np.array([
        [-half_length, -half_width],  # Bottom-left
        [half_length, -half_width],  # Bottom-right
        [half_length, half_width],  # Top-right
        [-half_length, half_width],  # Top-left
    ])

    # Rotation matrix to rotate the vehicle to its orientation
    cos_theta = np.cos(ego_orientation)
    sin_theta = np.sin(ego_orientation)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])

    # Rotate the corners based on the vehicle's orientation
    rotated_corners = (rotation_matrix @ corners.T).T

    # Translate the rotated corners to the vehicle's global position
    translated_corners = rotated_corners + ego_pos

    # Plot the vehicle as a polygon
    vehicle_polygon = Polygon(translated_corners, closed=True, edgecolor=color, facecolor='none', linewidth=2, label=label)

    plt.fill(vehicle_polygon.get_xy()[:, 0], vehicle_polygon.get_xy()[:, 1], color=color, label=label, zorder=zorder, **kwargs)


# MWE
if __name__ == '__main__':

    mpl.use('TkAgg')
    # Example parameters
    ego_pos = np.array([10.0, 5.0])
    ego_orientation = np.deg2rad(30)  # Convert 30 degrees to radians
    ego_dimensions = (4.5, 2.0)  # Length and width of the ego vehicle

    plot_ego_vehicle(ego_pos, ego_orientation, ego_dimensions)

    pedestrian_mean = np.array([12.0, 6.0])
    pedestrian_cov = np.array([[0.5, 0.1],
                               [0.1, 0.3]])

    # Analytical collision probability (without object size)
    analytical_prob = compute_collision_probability(ego_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov)
    print(f"Analytical Collision Probability: {analytical_prob}")

    # Monte Carlo collision probability (without object size)
    monte_carlo_prob = monte_carlo_collision_probability(
        ego_pos, ego_orientation, ego_dimensions, pedestrian_mean, pedestrian_cov, num_samples=10000, plot=True
    )
    print(f"Monte Carlo Collision Probability: {monte_carlo_prob}")

print('Done')