__author__ = "Korbinian Moller, Truls Nyberg"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import mapping, Polygon
from rasterio.features import geometry_mask
from multiprocessing import Pool, cpu_count, Manager

from commonroad.scenario.lanelet import LaneletType
from commonroad.scenario.obstacle import DynamicObstacle
import pedestrian_simulator.utils.helper_functions as hf
import pedestrian_simulator.utils.visualization as vis


class Policy:
    def __init__(self, scenario_id, lanelet_network, static_obstacles,
                 transform, grid_height, grid_width, max_action_length,
                 max_iterations=500, multiproc=False, plot=False):

        self.scenario_id = scenario_id
        self.static_obstacles = static_obstacles
        self.lanelet_network = lanelet_network
        self.static_obstacles = static_obstacles
        self.transform = transform
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_action_length = max_action_length

        self.plot = plot
        self.max_iterations = max_iterations
        self.multiproc = multiproc
        self._policies, self._costs_to_go, self._state_costs = self._generate_pedestrian_policies(use_multiprocessing=self.multiproc)

    @property
    def policies(self):
        return self._policies

    @property
    def costs_to_go(self):
        return self._costs_to_go

    @property
    def state_costs(self):
        return self._state_costs
           
    def get_dir(self, positions, policy_ids):
        inv_transform = ~self.transform
        x, y = inv_transform * (positions[:, 0], positions[:, 1])
        x0, y0 = np.floor(x - 0.5).astype(int), np.floor(y - 0.5).astype(int)

        desired_directions = self._policies[y0, x0, :, policy_ids]
        
        return desired_directions[:, [1, 0]] * [1, -1]

    @staticmethod
    def get_destination(ped: DynamicObstacle):
        final_state = ped.prediction.trajectory.state_list.pop(-1)  # Destination is stored here, pop to remove it from trajectory
        destination = final_state.position
        return destination

    def get_policy_id(self, destination):               
        for idx in range(self.policies.shape[-1]):
            destination_cell = np.round(~self.transform * destination).astype(int)
            action_at_destination = self.policies[destination_cell[1], destination_cell[0], :, idx]
            if np.array_equal(action_at_destination, np.array([0, 0])):
                return idx
                
        print("Picking policy 0, no goal found at destination: ", destination)
        return 0

    def _generate_pedestrian_policies(self, use_multiprocessing=False, num_processes=None):

        # Get policy folder
        policy_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'policies')

        # create hash value for scenario
        policy_hash = str(self.grid_height) + str(self.grid_width) + str(self.max_iterations) + str(self.max_action_length)

        # create filenames
        policy_filename = str(self.scenario_id) + '_' + policy_hash + '-policy.npy'
        costs_to_go_filename = str(self.scenario_id) + '_' + policy_hash + '-costs-to-go.npy'  # ToDo Remove when not needed anymore
        state_costs_filename = str(self.scenario_id) + '_' + policy_hash + '-state_costs.npy'

        # create paths to files
        policy_file = os.path.join(policy_folder, policy_filename)
        costs_to_go_file = os.path.join(policy_folder, costs_to_go_filename)
        state_costs_file = os.path.join(policy_folder, state_costs_filename)

        # Create folder if not exist
        os.makedirs(policy_folder, exist_ok=True)

        try:  # Speed up when simulating same scenario multiple times
            destinations_policies = np.load(policy_file)
            costs_to_go = np.load(costs_to_go_file)
            state_costs = np.load(state_costs_file)
            print("Loaded precomputed policies.")
        except FileNotFoundError:
            # generate state costs (cost to be on a certain position on the grid)
            state_costs = self._generate_state_costs(self.lanelet_network, self.static_obstacles,
                                                     self.transform, self.grid_height, self.grid_width)

            # generate possible actions
            actions = self._generate_actions(self.max_action_length)

            # calculate costs for each possible action
            action_costs = [np.linalg.norm(np.array(action)) for action in actions]

            # find all possible destinations and transform them into grid coordinates
            destinations = self._find_pedestrian_destinations(self.transform, self.lanelet_network)
            destinations_policies = np.zeros((self.grid_height, self.grid_width, 2, len(destinations)), dtype=int)
            costs_to_go = np.full((self.grid_height, self.grid_width, len(destinations)), np.inf)

            # create progress bar (pbar) for tracking
            with tqdm(total=self.max_iterations * len(destinations),
                      desc=f"Computing policies for {len(destinations)} destinations") as pbar:

                # if multiprocessing is enabled, multiple tasks are performed in parallel
                if use_multiprocessing:
                    # get cpu number
                    if num_processes is None:
                        num_processes = cpu_count()

                    # create multiprocessing manager and queue
                    manager = Manager()
                    progress_queue = manager.Queue()

                    # store required arguments in argument list
                    args_list = [(idx, destination, self.grid_width, self.grid_height, state_costs, actions, action_costs,
                                  self.max_iterations, self.plot, progress_queue)
                                 for idx, destination in enumerate(destinations)]

                    # create multiprocessing pool and perform calculation
                    with Pool(processes=num_processes) as pool:
                        results = pool.map_async(self._compute_policy_wrapper, args_list)

                        while not results.ready():
                            while not progress_queue.empty():
                                pbar.update(progress_queue.get())

                    # store results in variable and assign to policy array
                    results = results.get()
                    for idx, cost_to_go, policy in results:
                        destinations_policies[:, :, :, idx] = policy
                        costs_to_go[:, :, idx] = cost_to_go

                # if multiprocessing is deactivated, calculate policies sequentially
                else:
                    for idx, destination in enumerate(destinations):
                        _, cost_to_go, policy = self._compute_policy_for_destination(idx, destination, self.grid_width,
                                                                                     self.grid_height, state_costs, actions,
                                                                                     action_costs, self.max_iterations, self.plot,
                                                                                     pbar)
                        destinations_policies[:, :, :, idx] = policy
                        costs_to_go[:, :, idx] = cost_to_go

            # save policy as numpy .npy file
            np.save(policy_file, destinations_policies)
            np.save(costs_to_go_file, costs_to_go)  # Todo Remove when not needed anymore
            np.save(state_costs_file, state_costs)
            print("Saved computed policies as " + policy_filename)

        return destinations_policies, costs_to_go, state_costs

    def _compute_policy_for_destination(self, idx, destination, grid_width, grid_height, state_costs, actions, action_costs,
                                        max_iterations, plot_on, progress=None, update_interval=10):

        policy = np.zeros((grid_height, grid_width, 2), dtype=int)
        cost_to_go = np.full((grid_height, grid_width), np.inf)
        cost_to_go[destination[1], destination[0]] = 0.0
        state_costs[destination[1], destination[0]] = 0.0
        inf_mask = state_costs == np.inf

        for i in range(max_iterations):
            cost_to_go, policy = self._bellman_update(cost_to_go, policy, state_costs, actions, action_costs, inf_mask)
            if isinstance(progress, tqdm):
                progress.update(1)
            elif progress is not None and (i + 1) % update_interval == 0:
                progress.put(update_interval)  # Update progress

        if plot_on:
            vis.plot_policy(self.grid_height, self.grid_width, cost_to_go, policy)

        return idx, cost_to_go, policy

    def _compute_policy_wrapper(self, args):
        return self._compute_policy_for_destination(*args)

    def _generate_state_costs(self, lanelet_network, static_obstacles, transform, grid_height, grid_width, plot=False):

        # Initialize with high cost
        state_costs = np.full((grid_height, grid_width), np.inf)

        # Get all shapely polygons from lanelet network
        lanelets_shapely_polygons = [lanelet.polygon.shapely_object for lanelet in lanelet_network.lanelets if
                                     LaneletType.SIDEWALK not in lanelet.lanelet_type and
                                     LaneletType.CROSSWALK not in lanelet.lanelet_type]
        sidewalks_shapely_polygons = [lanelet.polygon.shapely_object for lanelet in lanelet_network.lanelets if
                                      LaneletType.SIDEWALK in lanelet.lanelet_type]
        crosswalks_shapely_polygons = [lanelet.polygon.shapely_object for lanelet in lanelet_network.lanelets if
                                       LaneletType.CROSSWALK in lanelet.lanelet_type]
        static_obstacles_polygon = [Polygon(hf.calc_corner_points(obst.initial_state.position,
                                                                  obst.initial_state.orientation,
                                                                  obst.obstacle_shape))
                                    for obst in static_obstacles]

        # Convert Shapely polygons to GeoJSON-like feature dicts
        lanelets_geo = self.poly_to_geo(lanelets_shapely_polygons)
        sidewalks_geo = self.poly_to_geo(sidewalks_shapely_polygons)
        crosswalks_geo = self.poly_to_geo(crosswalks_shapely_polygons)
        static_obstacles_geo = self.poly_to_geo(static_obstacles_polygon)

        # Rasterize the polygons, and flip to get origin in lower left corner
        # 'all_touched=True' considers all pixels that touch the polygon, 'invert=True' to mark polygon areas as 'True'
        rasterized_lanelets = geometry_mask(lanelets_geo, transform=transform, invert=True, all_touched=True,
                                            out_shape=(grid_height, grid_width))

        state_costs[rasterized_lanelets] = 50  # Order here matters

        if sidewalks_geo:
            rasterized_sidewalks = geometry_mask(sidewalks_geo, transform=transform, invert=True, all_touched=False,
                                                 out_shape=(grid_height, grid_width))
            state_costs[rasterized_sidewalks] = 10  # Lanelets must be first and then overwritten

        # if crosswalks exist, they shall be accounted for in the offline policy
        if crosswalks_geo:
            rasterized_crosswalks = geometry_mask(crosswalks_geo, transform=transform, invert=True, all_touched=True,
                                                  out_shape=(grid_height, grid_width))
            state_costs[rasterized_crosswalks] = 20  # We mark crosswalks last to let them float over the edges a bit

        # if static obstacles exist, they shall be accounted for in the offline policy
        if static_obstacles_geo:
            rasterized_obstacles = geometry_mask(static_obstacles_geo, transform=transform, invert=True,
                                                 all_touched=True, out_shape=(grid_height, grid_width))
            state_costs[rasterized_obstacles] = np.inf

        if plot:
            plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
            plt.imshow(state_costs, origin='upper', cmap='viridis', extent=(-0.5, grid_width - 0.5, -0.5, grid_height - 0.5))
            plt.title('Rasterized Polygons')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.xticks(range(0, grid_width + 1, 300))  # Adjust the ticks as needed
            plt.yticks(range(0, grid_height + 1, 300))  # Adjust the ticks as needed
            plt.gca().set_aspect('equal', adjustable='box')

        return state_costs

    @staticmethod
    def _find_pedestrian_destinations(transform, lanelet_network):

        destinations = []
        sidewalks = [lanelet for lanelet in lanelet_network.lanelets if LaneletType.SIDEWALK in lanelet.lanelet_type]

        for sidewalk in sidewalks:
            destinations.append(np.round(~transform * sidewalk.center_vertices[0]).astype(int))
            destinations.append(np.round(~transform * sidewalk.center_vertices[-1]).astype(int))

        return destinations

    @staticmethod
    def _bellman_update(costs_to_go, policy, state_costs, actions, action_costs, inf_mask=None):

        new_costs_to_go = costs_to_go.copy()
        new_policy = policy.copy()

        for dx, dy, action_cost in zip(*zip(*actions), action_costs):  # For each action
            possible_costs_to_go = np.roll(costs_to_go, shift=(dx, dy), axis=(0, 1))
            possible_state_costs = np.roll(state_costs * action_cost, shift=(dx, dy), axis=(0, 1))
            possible_costs_to_go = possible_costs_to_go + possible_state_costs

            # Update mask where cost is lower
            update_mask = possible_costs_to_go < new_costs_to_go

            # Apply updates
            new_costs_to_go[update_mask] = possible_costs_to_go[update_mask]
            new_policy[update_mask] = [-dx, -dy]

        if inf_mask is not None:
            # Make sure infinite cells remain infinite
            inf_array = np.full((inf_mask.shape[0], inf_mask.shape[1]), np.inf)
            new_costs_to_go[inf_mask] = inf_array[inf_mask]

        return new_costs_to_go, new_policy

    @staticmethod
    def _generate_actions(max_action_length):

        unique_angles = set()
        first_quadrant_actions = []
        for y in range(max_action_length):
            for x in range(max_action_length):
                if (x, y) == (0, 0):
                    continue  # Skip the origin
                angle = np.arctan2(y, x)
                if angle not in unique_angles:
                    unique_angles.add(angle)  # Mark this angle as seen
                    first_quadrant_actions.append((x, y))  # Add the point if the angle is unique
        # Mirror the points to cover all four quadrants
        all_actions = first_quadrant_actions.copy()
        all_actions += [(-x, y) for x, y in first_quadrant_actions]
        all_actions += [(x, -y) for x, y in first_quadrant_actions]
        all_actions += [(-x, -y) for x, y in first_quadrant_actions]

        return all_actions

    @staticmethod
    def poly_to_geo(polygons):
        return [mapping(polygon) for polygon in polygons]
