__author__ = "Truls Nyberg"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from typing import List
from functools import partial
from shapely import LineString, Point
from shapely import Polygon as ShapelyPolygon
from shapely.geometry import JOIN_STYLE
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.lanelet import Lanelet, LaneletType
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.geometry.shape import Polygon as CrPolygon
from commonroad.scenario.state import InitialState, ExtendedPMState
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory


# Create and save a new scenario file with sidewalks and pedestrians
def create_new_pedestrian_scenario(input_file, pedestrian_speed, pedestrians_per_cluster, position_deviation, cluster_distance, seed):
    
    # Set a seed for replication
    np.random.seed(seed)
    
    # Read scenario from file
    scenario, planning_problem_set = CommonRoadFileReader(input_file).open()

    # Set type to Urban if unknown to avoid warnings
    for lanelet in scenario.lanelet_network.lanelets:
        if not lanelet.lanelet_type:
            lanelet.lanelet_type.add(LaneletType.URBAN)

    # Find all beginning lanelets
    beginning_lanes = find_beginning_lanes(scenario)
    for lane in beginning_lanes:
        # Create sidewalks and check if they can be added without overlapping
        add_sidewalk_next_to_lane(lane, scenario)

    # Add pedestrians randomly on the sidewalks
    sidewalk_lanelets = [lanelet for lanelet in scenario.lanelet_network.lanelets if LaneletType.SIDEWALK in lanelet.lanelet_type]
    for sidewalk in sidewalk_lanelets:
        pedestrian_cluster_distances = sample_pedestrian_clusters(sidewalk.distance[-1], avg_distance_between_clusters=cluster_distance)
        for distance in pedestrian_cluster_distances:
            cluster_position, _, _, _ = sidewalk.interpolate_position(distance)
            pedestrian_positions = sample_pedestrians(cluster_position, avg_num_pedestrians=pedestrians_per_cluster, std_position_factor=position_deviation)
            destination = sample_destination(sidewalk_lanelets)
            for position in pedestrian_positions:
                pos_in_lane = scenario.lanelet_network.find_lanelet_by_position([position]) 
                if pos_in_lane != [[]]: # Skip spawning pedestrians outside the lane network
                    add_pedestrian(scenario, position, destination, walking_speed=pedestrian_speed)

    num_pedestrians = len(scenario.dynamic_obstacles)
    parameter_id = str(hash(tuple([pedestrian_speed, pedestrians_per_cluster, position_deviation, cluster_distance, seed])))[:5]
    old_scenario_id = str(scenario.scenario_id) 
    scenario.scenario_id.obstacle_behavior = 'P' # Not sure what P means, but stealing it to not get a warning..
    scenario.scenario_id.configuration_id = str(num_pedestrians)
    scenario.scenario_id.prediction_id = str(parameter_id)
    new_scenario_id = str(scenario.scenario_id)
    scenario.affiliation = 'Technical University of Munich'
    scenario.author = 'TUM Professorship Autonomous Vehicle Systems'
    scenario.source = 'AVS Pedestrian Simulator'
    
    print('Created new scenario', new_scenario_id, 'from', old_scenario_id, 'with', len(scenario.dynamic_obstacles), 'pedestrains.')
    
    # Return scenario, e.g., for plotting
    return scenario, planning_problem_set


# Find all beginning lanelets and merge them with all their successors
def find_beginning_lanes(scenario: Scenario):
    all_lanelets = scenario.lanelet_network.lanelets
    beginning_lanes = []
    beginning_lanelets = [lanelet for lanelet in all_lanelets if lanelet.predecessor == []]
    for lanelet in beginning_lanelets:
        successors = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, scenario.lanelet_network, 500)
        beginning_lanes.extend(successors[0])
    return beginning_lanes


# Add sidewalks next to lanes where possible
def add_sidewalk_next_to_lane(lane: Lanelet, scenario: Scenario, sidewalk_width = 2):
    
    lane_right_boundary = remove_duplicate_points(lane.right_vertices)
    lane_boundary_line = LineString(lane_right_boundary)

    sidewalk_left_linestring = lane_boundary_line
    sidewalk_center_linestring = lane_boundary_line.offset_curve(distance=-sidewalk_width/2, join_style=JOIN_STYLE.mitre)
    sidewalk_right_linestring = lane_boundary_line.offset_curve(distance=-sidewalk_width, join_style=JOIN_STYLE.mitre)
        
    sidewalk_left_polyline = np.array(sidewalk_left_linestring.coords)
    sidewalk_center_polyline = np.array(sidewalk_center_linestring.coords)
    sidewalk_right_polyline = np.array(sidewalk_right_linestring.coords)

    if not (len(sidewalk_left_polyline) == 
            len(sidewalk_center_polyline) == 
            len(sidewalk_right_polyline)):
        # Add extra points if one line has fewer points
        max_len = len(max(sidewalk_left_polyline, sidewalk_center_polyline, sidewalk_right_polyline, key=len))
        sidewalk_left_polyline = add_points(sidewalk_left_polyline, max_len)
        sidewalk_center_polyline = add_points(sidewalk_center_polyline, max_len)
        sidewalk_right_polyline = add_points(sidewalk_right_polyline, max_len)

    sidewalk = Lanelet(left_vertices=sidewalk_left_polyline, 
                    center_vertices=sidewalk_center_polyline, 
                    right_vertices=sidewalk_right_polyline,
                    lanelet_id=scenario.generate_object_id(), 
                    lanelet_type=set([LaneletType.SIDEWALK]))
    
    # Check overlap with shrunken sidewalk to avoid overlap with neighbour
    shapely_shrunk_sidewalk = sidewalk.polygon._shapely_polygon.buffer(-0.1)
    cr_shrunk_sidewalk = CrPolygon(shapely_shrunk_sidewalk.exterior.coords)
    overlapping_lanelets = scenario.lanelet_network.find_lanelet_by_shape(cr_shrunk_sidewalk)
    if overlapping_lanelets == []:
        scenario.add_objects([sidewalk])

# Add points when sidewalk boundaries are unequal in len
def add_points(coords, max_n):
    if len(coords) >= max_n:
        return coords
    new_coords = list(coords)
    # Interpolate between points and add points
    for i in range(max_n - len(coords)):
        start = coords[i%len(coords)]
        end = coords[(i + 1)%len(coords)]
        new_point = start + (end - start) / 2 
        new_coords.insert((1+i+i)%len(coords), new_point)
    return np.array(new_coords)

# Remove points close to each other to create valid boundaries
def remove_duplicate_points(points, decimals=0):
    seen = set()
    unique_points = []
    for point in points:
        # Round the point to the specified number of decimal places
        rounded_point = tuple(np.round(point, decimals))
        if rounded_point not in seen:
            seen.add(rounded_point)
            unique_points.append(point)
    return np.array(unique_points)

# Add pedestrians
def add_pedestrian(scenario: Scenario, start_pos: np.ndarray, goal_pos: np.ndarray, walking_speed=1.5, dt=0.1):
    ped_shape = Rectangle(0.5, 1)
    direction = (goal_pos - start_pos)
    orientation = math.atan2(direction[1], direction[0])
    trajectory = Trajectory(1, [ExtendedPMState(time_step=int(1), position=start_pos + walking_speed*direction*dt/np.linalg.norm(direction), velocity=walking_speed, orientation=orientation), 
                                ExtendedPMState(time_step=int(20), position=goal_pos, velocity=walking_speed, orientation=orientation)])   
    pedestrian = DynamicObstacle(obstacle_id=scenario.generate_object_id(), 
                                 obstacle_type=ObstacleType.PEDESTRIAN,
                                 obstacle_shape=ped_shape,
                                 initial_state=InitialState(time_step=0, position=start_pos, velocity=walking_speed, orientation=orientation),
                                 prediction=TrajectoryPrediction(trajectory=trajectory, shape=ped_shape))
    scenario.add_objects([pedestrian])

def sample_pedestrian_clusters(sidewalk_length, avg_distance_between_clusters=20):
    cluster_positions = []
    current_pos = np.random.exponential(scale=avg_distance_between_clusters)
    while current_pos < sidewalk_length:
        cluster_positions.append(current_pos)
        distance = np.random.exponential(scale=avg_distance_between_clusters)
        current_pos += distance
    return cluster_positions

def sample_pedestrians(cluster_center, avg_num_pedestrians=2.0, std_position_factor=0.2):
    num_pedestrians = int(np.random.exponential(scale=avg_num_pedestrians - 1.0)) + 1  # At least 1 pedestrian (actually geometric distribution)
    pedestrian_positions = np.random.normal(loc=cluster_center, 
                                            scale=(std_position_factor*num_pedestrians, std_position_factor*num_pedestrians), 
                                            size=(num_pedestrians,2))
    return pedestrian_positions

def sample_destination(sidewalks:List[Lanelet]):
    target_sidewalk = np.random.choice(sidewalks)
    destination = np.random.choice([0, -1])
    return target_sidewalk.center_vertices[destination]

def closest_point_on_closest_polygon(point, polygons):
    """
    Finds the closest polygon to a given point from a list of Shapely polygons.

    :param point: The point in space (as a tuple or list, e.g., [x, y]).
    :param polygons: A list of Shapely Polygon objects.
    :return: The closest point on that polygon.
    """
    point = Point(point)
    min_distance = float('inf')
    closest_point_on_polygon = None

    for polygon in polygons:
        distance = polygon.distance(point)
        if distance < min_distance:
            min_distance = distance
            closest_point_on_polygon = polygon.exterior.interpolate(polygon.exterior.project(point))

    return closest_point_on_polygon

def on_click(event, scenario: Scenario, fig, rnd: MPRenderer):
    # This function is called when the plot is clicked
    if event.inaxes is None:
        return

    x, y = event.xdata, event.ydata

    clicked_lanelets_list = scenario.lanelet_network.find_lanelet_by_position([np.array([x,y])])
    clicked_lanelet_ids = clicked_lanelets_list[0]
    
    clicked_lanelets = [scenario.lanelet_network.find_lanelet_by_id(lanelet_id) for lanelet_id in clicked_lanelet_ids]
    clicked_crosswalks = [lanelet for lanelet in clicked_lanelets if LaneletType.CROSSWALK in lanelet.lanelet_type]
    clicked_sidewalks = [lanelet for lanelet in clicked_lanelets if LaneletType.SIDEWALK in lanelet.lanelet_type]

    if clicked_crosswalks:
        scenario.remove_lanelet(clicked_crosswalks)
        print("Crosswalk removed!")

    if clicked_sidewalks == [] and clicked_crosswalks == []:
        print("Click on a sidewalk to add a crosswalk!")
        print("Click on a crosswalk to remove it!")
    
    for sidewalk in clicked_sidewalks:
        point = Point([x,y])
        sidewalk_boundary = LineString(sidewalk.left_vertices)
        center_point = sidewalk_boundary.interpolate(sidewalk_boundary.project(point))
        right_point = sidewalk_boundary.interpolate(sidewalk_boundary.project(point) - 2)
        left_point = sidewalk_boundary.interpolate(sidewalk_boundary.project(point) + 2)

        other_sidewalk_polygons = [lanelet.polygon._shapely_polygon 
                                for lanelet in scenario.lanelet_network.lanelets 
                                if (LaneletType.SIDEWALK in lanelet.lanelet_type 
                                    and lanelet.lanelet_id != sidewalk.lanelet_id)]
        center_point_opp = closest_point_on_closest_polygon(center_point, other_sidewalk_polygons)
        right_point_opp = closest_point_on_closest_polygon(right_point, other_sidewalk_polygons)
        left_point_opp = closest_point_on_closest_polygon(left_point, other_sidewalk_polygons)

        crossing_center_vertices = np.array([point.xy for point in [center_point, center_point_opp]]).squeeze()
        crossing_right_vertices = np.array([point.xy for point in [right_point, right_point_opp]]).squeeze()
        crossing_left_vertices = np.array([point.xy for point in [left_point, left_point_opp]]).squeeze()

        crosswalk = Lanelet(left_vertices=crossing_left_vertices, 
                            center_vertices=crossing_center_vertices, 
                            right_vertices=crossing_right_vertices, 
                            lanelet_id=scenario.generate_object_id(), 
                            lanelet_type=set([LaneletType.CROSSWALK]))
        scenario.add_objects([crosswalk])
        print("Crosswalk added!")
    
    plt.figure(fig)
    scenario.draw(rnd)
    rnd.render(show=True)


if __name__ == "__main__":
    # Input scenario file path
    input_file = "../../example_scenarios/ZAM_Tjunction/ZAM_Tjunction-1_42_T-1.xml"

    # Visualization settings
    mpl.use('TkAgg')  # Use Tkinter as the backend for interactive plots

    # Parameters for pedestrian scenario generation
    pedestrian_speed = 1.2  # Average speed of pedestrians (m/s)
    pedestrians_per_cluster = 2.0  # Number of pedestrians per cluster
    position_deviation = 0.2  # Deviation in cluster positions
    cluster_distance = 4.0  # Distance between clusters
    seed = 30  # Random seed for reproducibility

    # Create a new pedestrian scenario by extending the input scenario
    scenario, planning_problem_set = create_new_pedestrian_scenario(
        input_file,
        pedestrian_speed,
        pedestrians_per_cluster,
        position_deviation,
        cluster_distance,
        seed
    )

    # Set up visualization
    fig = plt.figure(figsize=(25, 10))
    rnd = MPRenderer()

    # Customize drawing parameters
    rnd.draw_params.dynamic_obstacle.draw_icon = True
    rnd.draw_params.lanelet_network.lanelet.show_label = True

    # Draw the scenario and planning problems
    scenario.draw(rnd)
    planning_problem_set.draw(rnd)
    rnd.render(show=True)

    # Interactive event handler for mouse clicks
    fig.canvas.mpl_connect('button_press_event', partial(on_click, scenario=scenario, fig=fig, rnd=rnd))

    # Display the plot
    plt.show(block=True)

    # Save the updated scenario to a new XML file
    output_file = f"../../example_scenarios/generated_scenarios/{scenario.scenario_id}.xml"  # Enter valid output path
    fw = CommonRoadFileWriter(scenario=scenario, planning_problem_set=planning_problem_set)
    fw.write_to_file(output_file, OverwriteExistingFile.ASK_USER_INPUT)
    print(f"Scenario saved to {output_file}")
