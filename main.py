__author__ = "Korbinian Moller, Truls Nyberg"
__copyright__ = "TUM Professorship Autonomous Vehicle Systems"
__version__ = "1.0"
__maintainer__ = "Korbinian Moller"
__email__ = "korbinian.moller@tum.de"
__status__ = "Beta"

# imports
import matplotlib as mpl
import matplotlib.pyplot as plt

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

from pedestrian_simulator.pedestrian_simulator import PedestrianSimulator

if __name__ == '__main__':

    # Specify the path to the scenario file. This file contains the initial setup of the simulation environment.
    scenario_path = 'example_scenarios/DEU_Ffb/DEU_Ffb-1_230_P--2202.xml'

    # Simulation settings
    sim_steps = 100  # Number of simulation steps to perform
    animate = True  # Whether to animate the simulation in real time
    save_scenario = False  # Whether to save the modified scenario after simulation
    plot_limits = [-40, 100, -40, 40]  # Plot limits for visualization

    # Use an alternate backend for matplotlib if animation is enabled.
    if animate:
        mpl.use('TkAgg')

    # Load the scenario and the corresponding planning problem set from the file.
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()

    # Initialize the pedestrian simulator with the loaded scenario.
    pedestrian_simulator = PedestrianSimulator(scenario)

    # Create a renderer for visualization with specified figure size and plot limits.
    rnd = MPRenderer(figsize=(60, 40), plot_limits=plot_limits)

    # Configure plotting parameters for the visualization.
    params = MPDrawParams()
    params.dynamic_obstacle.draw_icon = True  # Draw icons for dynamic obstacles
    params.dynamic_obstacle.draw_shape = False  # Do not draw the shape of dynamic obstacles
    params.dynamic_obstacle.draw_bounding_box = False  # Do not draw bounding boxes of dynamic obstacles
    params.dynamic_obstacle.show_label = False  # Do not show labels for dynamic obstacles

    # Main simulation loop
    for t in range(sim_steps):

        # Advance the simulation by one step, updating pedestrian states.
        pedestrian_simulator.step_pedestrians(t)

        # If animation is enabled, render the current state of the scenario.
        if animate:
            params.time_begin = t  # Set the time step for visualization
            scenario.draw(rnd, draw_params=params)  # Draw the scenario
            rnd.render(show=True)  # Render the plot
            plt.pause(0.1)  # Pause briefly to create animation effect

    # Final visualization of the scenario at the last simulation step.
    params.time_begin = 1  # Set the time step to the first frame -> visualize trajectory
    scenario.draw(rnd, draw_params=params)  # Draw the final scenario state
    rnd.render(show=True)  # Render the final plot
    plt.show()  # Display the visualization

    # Save the updated scenario to a new file if enabled.
    if save_scenario:
        fw = CommonRoadFileWriter(scenario=scenario, planning_problem_set=planning_problem_set)  # Initialize the writer
        fw.write_to_file(f"{scenario.scenario_id}_simulated.xml", OverwriteExistingFile.ASK_USER_INPUT)  # Write to file
        print('Scenario saved')  # Confirm the scenario was saved