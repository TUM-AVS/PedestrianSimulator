__author__ = "Korbinian Moller, Truls Nyberg"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import List
from pedestrian_simulator.pysocialforce.forces import Force


def plot_policy(grid_height, grid_width, policy, idx=0, additional_map=None, interactive=False, fig=None, transform=None):
    """
    Plot the policy directions and cost-to-go values for a given policy array and cost-to-go array.

    Args:
        grid_height (int): The height of the grid.
        grid_width (int): The width of the grid.
        additional_map (numpy.ndarray or None): A 3D numpy array representing additional values for each state (cost-to-go, state_costs).
        policy (numpy.ndarray): A 4D numpy array representing the policy for each state.
        idx (int): The index of the policy to plot.
        interactive (bool): Whether to display the plot interactively.
        fig (matplotlib.figure.Figure): A matplotlib figure to plot on.
        transform (Affine): An affine transformation to apply to the grid
    """

    # Use interactive mode if specified
    if interactive:
        mpl.use('TkAgg')

    # Create a new figure and axis if none are provided
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        ax = fig.axes[0]

    # Create meshgrid for X and Y coordinates
    x_grid, y_grid = np.meshgrid(np.arange(grid_width), np.arange(grid_height))

    # Flip the policy array upside down and normalize the direction vectors
    if len(policy.shape) == 4:
        directions_array = np.flipud(policy[:, :, :, idx])
    else:
        directions_array = np.flipud(policy)

    norms = np.linalg.norm(directions_array, axis=2, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero
    normalized_directions = directions_array / norms

    # Transform coordinates and directions if a transformation matrix is provided
    if transform is not None:
        # Convert the affine transformation to a 3x3 matrix
        transform_matrix = np.array([
            [transform.a, transform.b, transform.c],
            [transform.d, transform.e, transform.f],
            [0, 0, 1]])

        # Invert the Y coordinates before transformation
        y_grid = np.flipud(y_grid)

        # Apply the affine transformation to the grid coordinates
        grid_coords = np.vstack([x_grid.flatten(), y_grid.flatten(), np.ones_like(x_grid.flatten())])
        transformed_coords = transform_matrix @ grid_coords
        x_transformed = transformed_coords[0, :].reshape(x_grid.shape)
        y_transformed = transformed_coords[1, :].reshape(y_grid.shape)

        # Calculate extent for imshow
        extent = (x_transformed.min(), x_transformed.max(), y_transformed.min(), y_transformed.max())

    else:
        # Use the original coordinates if no transformation is provided
        x_transformed = x_grid
        y_transformed = y_grid
        extent = (-0.5, grid_width - 0.5, -0.5, grid_height - 0.5)

    u_transformed = normalized_directions[:, :, 1]
    v_transformed = -normalized_directions[:, :, 0]

    # Create a mask to filter out zero vectors
    mask = (u_transformed != 0) | (v_transformed != 0)

    # Display the cost_to_go image
    if additional_map is not None:
        # show costs to go
        if len(additional_map.shape) == 3:
            cax = ax.imshow(additional_map[:, :, idx], origin='upper', cmap='viridis', extent=extent, zorder=100, alpha=0.6)
            label = 'Cost-to-go'

        # show state_costs
        else:
            cax = ax.imshow(additional_map, origin='upper', cmap='viridis', extent=extent, zorder=100, alpha=0.6)
            label = 'State Costs'

        # show additional map
        fig.colorbar(cax, ax=ax, label=label, orientation='vertical')

    # Plot the quiver plot for the policy directions
    ax.quiver(x_transformed[mask], y_transformed[mask], u_transformed[mask], v_transformed[mask], color='white',
              scale=100, headwidth=1, zorder=100)
    ax.set_title('Optimal Policy')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def plot_forces(peds, forces : List[Force], interactive, fig=None):

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        ax = fig.axes[0]

    colors = ['red', 'green', 'blue', 'yellow', 'brown', 'black']
    positions = peds.pos()
    for color_idx, force in enumerate(forces):
        force_vectors = force.get_force()
        label = force.__class__.__name__
        for pos_idx, pos in enumerate(positions):
            if pos_idx == 0:
                plot_vector(ax, pos, force_vectors[pos_idx, :], colors[color_idx], label)
            else:
                plot_vector(ax, pos, force_vectors[pos_idx, :], colors[color_idx], label=None)

    ax.legend()


def plot_vector(ax: Axes, point, vector, color, label):
    """
    Plots a vector on a given Matplotlib axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot the vector on.
    point (array-like): The starting point of the vector.
    vector (array-like): The vector to plot.
    **kwargs: Additional keyword arguments to pass to ax.plot.
    """
    point = np.array(point)
    vector = np.array(vector)

    ax.arrow(point[0], point[1], vector[0], vector[1],
             head_width=0.1, head_length=0.2, fc=color, ec=color, zorder=100, label=label)


def plot_vehicle_prediction(predicted_paths, fig):

    ax = fig.axes[0]
    for path in predicted_paths:
        ax.plot(path[:, 0], path[:, 1], zorder=100, color='red')
