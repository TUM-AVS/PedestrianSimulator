__author__ = "Korbinian Moller, Truls Nyberg"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from rasterio.transform import from_origin
import math
from omegaconf import OmegaConf

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.lanelet import Lanelet
from commonroad.common.util import subtract_orientations

from pedestrian_simulator.utils.policy import Policy

from pedestrian_simulator.pysocialforce.utils import DefaultConfig
from pedestrian_simulator.pysocialforce.scene import PedState, EnvState
from pedestrian_simulator.pysocialforce import forces
import pedestrian_simulator.utils.visualization as vis


class PedestrianSimulator:
    """
    A simulator for modeling pedestrian behavior using a social force model.

    This class integrates with the CommonRoad framework to simulate and predict pedestrian movements in a scenario.
    It uses a policy-based approach for destination determination and allows for visualization, prediction, and interaction
    with vehicles and obstacles.

    Attributes:
        scenario (Scenario): The CommonRoad scenario object containing all relevant simulation information.
        pedestrian_sim_config (OmegaConf): Configuration parameters for the pedestrian simulator.
        config (DefaultConfig): Configuration for the social force model.
        delta_t (float): Time step for the simulation.
        scene_config (OmegaConf): Scene-specific configuration parameters.
        scale_factor_length (float): Scale factor for pedestrian shape length.
        scale_factor_width (float): Scale factor for pedestrian shape width.
        pedestrian_predictions (dict): Stores predictions of pedestrian trajectories and states.
        max_action_length (int): Maximum allowed action length in the policy.
        grid_cell_size (float): Size of the grid cells used for state discretization.
        max_iterations (int): Maximum number of iterations for the Bellman update algorithm.
        plot (bool): Flag to enable or disable debug plotting.
        transform (Affine): Affine transformation defining the grid layout for the environment.
        policy (Policy): Instance of the Policy class for managing offline policies.
        prediction_horizon (int): Time horizon for pedestrian and vehicle prediction.
        peds (PedState): State representation of all pedestrians.
        env (EnvState): State representation of the environment, including obstacles and vehicles.
        vehicles (list): List of vehicle objects in the scenario.
        predicted_vehicle_paths (list): List of predicted paths for vehicles.
        forces (list): List of forces acting on the pedestrians, derived from the social force model.

    Methods:
        initialize_ped_state_from_cr(): Initializes pedestrian states from the CommonRoad scenario.
        update_ped_state_to_commonroad(timestep): Updates pedestrian states in the CommonRoad scenario representation.
        update_env_state_from_cr(timestep): Updates the environment state based on the scenario at a given timestep.
        update_dir_from_policy(): Updates pedestrian movement directions using the policy.
        step_pedestrians(timestep): Advances the pedestrian simulation by one timestep.
        make_forces(force_configs): Constructs the forces acting on pedestrians.
        compute_forces(): Calculates the combined forces acting on the pedestrians.
        predict_vehicle(position, speed, orientation): Predicts vehicle paths based on their position, speed, and orientation.
        prediction_along_centerline(center_line, start_point, distance): Generates predictions along a lane's centerline.
        closest_point_on_centerline(center_line, start_point): Finds the closest point on a lane's centerline to a given position.
        path_to_obstacles(path): Converts a path into obstacle representations.
        debug_plot(): Visualizes policies, forces, and vehicle predictions for debugging.
        create_pedestrian_predictions(): Generates pedestrian trajectory predictions.
        _rotation_matrix(angle): Returns a 2D rotation matrix for a given angle.
        _create_rot_cov_matrix(pos_list, orientation, initial_variance, ...): Creates rotated covariance matrices for predicted trajectories.
        _get_grid_transform(): Computes the grid transform and its dimensions for the simulation.
        _load_default_config(): Loads the default configuration for the pedestrian simulator.
    """

    def __init__(self, scenario: Scenario, pedestrian_sim_config=None, social_force_config=None):

        # Check and load pedestrian simulator configuration
        if pedestrian_sim_config:
            # If a configuration is provided, use it
            self.pedestrian_sim_config = pedestrian_sim_config
        else:
            # Otherwise, load the default configuration
            self.pedestrian_sim_config = self._load_default_config()

        # Ensure the pedestrian simulator is enabled in the configuration
        if not self.pedestrian_sim_config.use_pedestrian_simulator:
            raise ValueError(
                "Pedestrian simulator is not enabled in the configuration file. "
                "This state indicates a logical error in the configuration flow."
            )

        # Reference the same scenario as used outside
        self.scenario: Scenario = scenario
        self.delta_t = scenario.dt  # Time step for simulation

        # Load the social force model configuration
        self.config = DefaultConfig()
        if social_force_config:
            self.config.load_config(social_force_config)
        self.scene_config = self.config.sub_config("scene")

        # Ego prediction parameters
        self.scale_factor_length = self.pedestrian_sim_config.scale_factor_length
        self.scale_factor_width = self.pedestrian_sim_config.scale_factor_width
        self.pedestrian_predictions = {}
      
        # Parameters for offline policy for Desired Destination
        self.max_action_length = self.pedestrian_sim_config.max_action_length  # Sets action discretization
        self.grid_cell_size = self.pedestrian_sim_config.grid_cell_size  # Sets state discretization
        self.max_iterations = self.pedestrian_sim_config.max_iterations  # Max iterations of the bellman update
        self.plot = False  # activate or deactivate debug plot

        self.transform, grid_height, grid_width = self._get_grid_transform()

        # create Policy class which manages the offline policies
        self.policy = Policy(scenario_id=self.scenario.scenario_id,
                             lanelet_network=self.scenario.lanelet_network,
                             static_obstacles=self.scenario.static_obstacles,
                             transform=self.transform,
                             grid_height=grid_height,
                             grid_width=grid_width,
                             max_action_length=self.max_action_length,
                             max_iterations=self.max_iterations,
                             plot=self.plot,
                             multiproc=self.pedestrian_sim_config.use_multiprocessing)
      
        # Pedestrian vehicle prediction horizon
        self.prediction_horizon = self.pedestrian_sim_config.prediction_horizon

        # initiate agents
        self.peds = None
        self.initialize_ped_state_from_cr()

        # initiate obstacles
        self.env = None
        self.vehicles = None
        self.predicted_vehicle_paths = None
        self.update_env_state_from_cr(timestep=0)

        # construct forces
        self.forces = self.make_forces(self.config)

    def initialize_ped_state_from_cr(self):
        pedestrians = [dyn_obs for dyn_obs in self.scenario.dynamic_obstacles if dyn_obs.obstacle_type == ObstacleType.PEDESTRIAN]
        
        # state: (x, y, v_x, v_y, goal_x, goal_y, policy_id, obstacle_id, dir_x, dir_y)
        ped_state = np.zeros((len(pedestrians), 10)) 

        for idx, ped in enumerate(pedestrians):
            velocity = ped.initial_state.velocity
            orientation = ped.initial_state.orientation
            vx = velocity * np.cos(orientation)
            vy = velocity * np.sin(orientation)
            destination = self.policy.get_destination(ped)
            policy_id = self.policy.get_policy_id(destination)

            ped_state[idx, 0:2] = ped.initial_state.position
            ped_state[idx, 2:4] = np.array([vx, vy])
            ped_state[idx, 4:6] = destination
            ped_state[idx, 6] = policy_id
            ped_state[idx, 7] = ped.obstacle_id
        
        groups = []
        self.peds = PedState(ped_state, groups, self.config)     

    def update_ped_state_to_commonroad(self, timestep):
        obstacle_ids = self.peds.obstacle_id()
        positions = self.peds.pos()
        velocity_vectors = self.peds.vel()
        orientations = np.arctan2(velocity_vectors[:, 1], velocity_vectors[:, 0])

        goal_positions = self.peds.goal()
        dist_to_goal = np.linalg.norm(goal_positions - positions, axis=1)
        at_goal_mask = dist_to_goal < 1

        self.peds.update(state=self.peds.state[~at_goal_mask], groups=[])
        for i, id in enumerate(obstacle_ids):
            cr_ped = self.scenario.obstacle_by_id(id)
            
            # Update the state. Copy initial state if none exists
            cr_state = cr_ped.state_at_time(timestep)
            if cr_state is None:
                cr_state = CustomState(time_step=timestep)
                cr_ped.prediction.trajectory.state_list.append(cr_state)

            cr_state.position = positions[i, :]
            cr_state.velocity = np.linalg.norm(velocity_vectors[i, :])
            cr_state.orientation = orientations[i]
        
        for id in obstacle_ids[at_goal_mask]:
            cr_obstacle = self.scenario.obstacle_by_id(id)
            self.scenario.remove_obstacle(cr_obstacle)

    def update_env_state_from_cr(self, timestep):
        self.vehicles = [dyn_obs for dyn_obs in self.scenario.dynamic_obstacles if dyn_obs.obstacle_type != ObstacleType.PEDESTRIAN]
        obstacles = [] if self.vehicles else None
        self.predicted_vehicle_paths = []
        for cr_vehicle in self.vehicles:
            cr_state = cr_vehicle.state_at_time(timestep)
            if cr_state is None:
                continue

            position = cr_state.position
            speed = cr_state.velocity
            heading = cr_state.orientation
            direction = np.array([np.cos(heading), np.sin(heading)])
            offset = (cr_vehicle.obstacle_shape.length - cr_vehicle.obstacle_shape.width)/2
            rear_pos = position - offset * direction
            
            predicted_paths = self.predict_vehicle(rear_pos, speed, heading)
            self.predicted_vehicle_paths.extend(predicted_paths)
            for path in predicted_paths:
                obstacle_lines = self.path_to_obstacles(path)
                obstacles.extend(obstacle_lines)

        self.env = EnvState(obstacles)

    def update_dir_from_policy(self):
        dir = self.policy.get_dir(self.peds.pos(), self.peds.policy_id())
        norm = np.linalg.norm(dir, axis=-1)
        norm = np.maximum(norm, 1e-8)
        norm_dir = dir / np.expand_dims(norm, axis=1)
        self.peds.update_dir(norm_dir)

    def step_pedestrians(self, timestep):
        # Update previously planned
        self.update_ped_state_to_commonroad(timestep)
        
        # Plan next step
        self.update_env_state_from_cr(timestep)
        self.update_dir_from_policy()
        self.peds.step(self.compute_forces())

    def make_forces(self, force_configs):
        """Construct forces"""
        force_list = [
            forces.PedRepulsiveForce(),
            forces.GoalAttractiveForce(),
            # forces.DesiredForce(),
            # forces.SocialForce(),
            forces.VehicleForce()
        ]
        group_forces = [
            # forces.GroupCoherenceForceAlt(),
            # forces.GroupRepulsiveForce(),
            # forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs, self.delta_t)

        return force_list

    def compute_forces(self):
        return sum(map(lambda x: x.get_force(), self.forces))

    def predict_vehicle(self, position, speed, orientation):

        predicted_paths = []
        lanelet_ids_in = self.scenario.lanelet_network.find_lanelet_by_position([position])[0]

        if speed < 0.5:
            return predicted_paths
        
        lanelets_in_orientation = []
        for lanelet_id in lanelet_ids_in:
            lanelet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            lane_heading = lanelet.orientation_by_position(position)
            
            heading_diff = np.abs(subtract_orientations(lane_heading, orientation))

            if heading_diff <= np.pi / 4:
                lanelets_in_orientation.append(lanelet)
        
        # Assure the lanelets are long enough to make a prediction on
        for lanelet in lanelets_in_orientation:
            lanes, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, self.scenario.lanelet_network, max_length=speed*self.prediction_horizon)
            for ln in lanes:
                path = self.prediction_along_centerline(ln.center_vertices, position, speed*self.prediction_horizon)
                predicted_paths.append(path)
        return predicted_paths

    def prediction_along_centerline(self, center_line, start_point, distance):
        """Find points on the center line starting from the closest point to start_point and up to a certain distance."""
        n = len(center_line)

        # Step 1: Find the closest point on the center line
        closest_point, closest_segment, closest_t = self.closest_point_on_centerline(center_line, start_point)
        
        # Step 2: Accumulate points along the center line starting from the closest point
        points = [closest_point]
        cumulative_dist = 0
        segment_index = closest_segment

        # If the closest point was not at the beginning of the segment, calculate the partial distance
        if closest_t < 1:
            next_point = center_line[segment_index + 1]
            segment_length = np.linalg.norm(next_point - closest_point)
            cumulative_dist += segment_length
            points.append(next_point)
            segment_index += 1

        # Add full segments until we reach the distance limit
        while cumulative_dist < distance and segment_index < n - 1:
            next_point = center_line[segment_index + 1]
            segment_length = np.linalg.norm(center_line[segment_index] - next_point)
            if cumulative_dist + segment_length > distance:
                # Interpolate the final point
                remaining_distance = distance - cumulative_dist
                direction = (next_point - center_line[segment_index]) / segment_length
                final_point = center_line[segment_index] + direction * remaining_distance
                points.append(final_point)
                break
            else:
                points.append(next_point)
                cumulative_dist += segment_length
                segment_index += 1

        return np.array(points) 

    @staticmethod
    def closest_point_on_centerline(center_line, start_point):
        """Find the closest point on the centerline to the start_point."""
        a = center_line[:-1]  # Start points of segments
        b = center_line[1:]   # End points of segments
        ap = start_point - a  # Vector from a to start_point
        ab = b - a            # Vector from a to b

        # Projection of ap onto ab, expressed as a fraction of the length of ab
        t = np.sum(ap * ab, axis=1) / np.sum(ab * ab, axis=1)
        t = np.clip(t, 0, 1)  # Clamp t between 0 and 1 to stay within the segment

        # Compute the closest points on each segment
        closest_points = a + t[:, np.newaxis] * ab

        # Compute the distances from start_point to each closest point
        distances = np.linalg.norm(closest_points - start_point, axis=1)

        # Find the index of the segment with the minimum distance
        closest_index = np.argmin(distances)
        closest_point = closest_points[closest_index]
        
        return closest_point, closest_index, t[closest_index]

    @staticmethod
    def path_to_obstacles(path):
        obstacles = []
        for i in range(len(path) - 1):
            start_x, start_y = path[i]
            end_x, end_y = path[i+1]
            obstacles.append((start_x, end_x, start_y, end_y))
        return obstacles

    def debug_plot(self):

        mpl.use('TkAgg')
        fig = plt.gcf()
        ax = fig.gca()

        policies = self.policy.policies
        costs_to_go = self.policy.costs_to_go
        state_costs = self.policy.state_costs

        vis.plot_policy(self.policy.grid_height, self.policy.grid_width, policies, additional_map=None, interactive=True,
                        fig=fig, transform=self.transform)
        
        vis.plot_forces(self.peds, self.forces, interactive=True, fig=fig)

        vis.plot_vehicle_prediction(self.predicted_vehicle_paths, fig=fig)

        # plt.show()

    def create_pedestrian_predictions(self):

        # clear predictions
        self.pedestrian_predictions.clear()

        for pedestrian in self.peds.state:

            position = pedestrian[0:2]
            velocity = pedestrian[2:4]
            pedestrian_id = int(pedestrian[7])

            # buffer variables
            orientation = math.atan2(velocity[1], velocity[0])

            # Compute the number of required time steps
            num_steps = int(self.prediction_horizon / self.delta_t) + 1

            # Create a matrix of time steps and velocities
            t = np.arange(num_steps)[:, np.newaxis] * self.delta_t

            # Compute the trajectory
            pos_list = position + t * velocity.T

            # create dict like commonroad predictions e.g. walenet
            # create velocity list
            v_list = np.full(len(pos_list), np.linalg.norm(velocity))

            # create orientation list
            orientation_list = np.full(len(pos_list), orientation)

            # create shape with scale factor
            shape = {'length': 0.5 * self.scale_factor_length,
                     'width': 1.0 * self.scale_factor_width}

            # create simple covariance matrix
            cov_list = self._create_rot_cov_matrix(pos_list, orientation, initial_variance=0.1)

            # combine everything to dict
            self.pedestrian_predictions[pedestrian_id] = {'orientation_list': orientation_list,
                                                          'v_list': v_list,
                                                          'pos_list': pos_list,
                                                          'shape': shape,
                                                          'cov_list': cov_list}
        return self.pedestrian_predictions

    @staticmethod
    def _rotation_matrix(angle):
        """
        Returns a 2D rotation matrix for the given angle in radians.
        """
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

    def _create_rot_cov_matrix(self, pos_list, orientation, initial_variance, variance_factor_x=1.02,
                               variance_factor_y=1.08):
        """
        Creates a list of covariance matrices for each position in the trajectory,
        with an exponential increase in variance and rotation based on orientation.

        :param pos_list: List of positions (x, y) of the trajectory.
        :param orientation: Orientation (in radians) for the entire trajectory.
        :param initial_variance: Initial variance value.
        :param variance_factor_x: Factor by which variance increases exponentially for the x direction.
        :param variance_factor_y: Factor by which variance increases exponentially for the y direction.
        :return: Array of covariance matrices.
        """
        # Number of positions in the trajectory
        num_positions = len(pos_list)

        # Calculate exponential variances for all positions in x and y directions
        variances_x = initial_variance * np.power(variance_factor_x, np.arange(num_positions))
        variances_y = initial_variance * np.power(variance_factor_y, np.arange(num_positions))

        # Rotation matrix for the given orientation
        rot_matrix = self._rotation_matrix(orientation)

        # Create covariance matrices with different variances in X and Y
        cov_matrices = []
        for var_x, var_y in zip(variances_x, variances_y):
            # Initial covariance matrix
            cov_matrix = np.array([[var_x, 0], [0, var_y]])

            # Rotate the covariance matrix
            rotated_cov_matrix = rot_matrix @ cov_matrix @ rot_matrix.T
            cov_matrices.append(rotated_cov_matrix)

        return np.array(cov_matrices)

    def _get_grid_transform(self):
        # Find size of grid and add buffer around to not roll over in value iteration
        max_coords = np.max([polygon._max for polygon in self.scenario.lanelet_network.lanelet_polygons], axis=0)
        min_coords = np.min([polygon._min for polygon in self.scenario.lanelet_network.lanelet_polygons], axis=0)

        west = min_coords[0] - self.max_action_length*self.grid_cell_size
        north = max_coords[1] + self.max_action_length*self.grid_cell_size
        transform = from_origin(west=west, north=north, xsize=self.grid_cell_size, ysize=self.grid_cell_size)
        
        grid_size = np.ceil((max_coords - min_coords) / self.grid_cell_size).astype(int)
        grid_width, grid_height = grid_size + 2*self.max_action_length 

        return transform, grid_height, grid_width

    @staticmethod
    def _load_default_config():

        # loads the config.yaml file
        filepath = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        print("Load default pedestrian simulation settings!")

        with open(filepath, 'r') as file:
            config = OmegaConf.load(filepath)

        return config
