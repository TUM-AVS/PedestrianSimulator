<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13493227.svg)](https://zenodo.org/records/tbd) -->

[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)


# Pedestrian Simulator for Social Force Modeling

This repository includes a Pedestrian Simulation Framework built on the CommonRoad scenario format. It simulates pedestrian behavior using a social force model and integrates policy-based movement planning for dynamic and realistic pedestrian predictions.

For a practical application of this pedestrian simulation model, explore our [Pedestrian Aware Motion Planner](https://github.com/TUM-AVS/PedestrianAwareMotionPlanning).

## Features

- **Social Force Model**: Simulates pedestrian interactions and behaviors.
- **Policy-based Planning**: Implements movement policies for destination selection.
- **Integration with CommonRoad**: Direct compatibility with CommonRoad scenarios.
- **Visualization**: Detailed rendering of scenarios and pedestrian trajectories.


---

## 🔧 Requirements & Installation

<details>
<summary>Click to expand</summary>

### Requirements
The software is developed and tested on recent versions of Linux. We strongly recommend using [Ubuntu 22.04](https://ubuntu.com/download/desktop) or higher.
For the Python installation, we suggest the usage of Virtual Environment with Python 3.11, Python 3.10 or Python 3.9
For the development IDE we suggest [PyCharm](http://www.jetbrains.com/pycharm/)

### 1. Clone the repository

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

### 2. Create and activate a new Virtual Environment

   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

### 3. Install all required packages


#### Installation with Poetry
To install the project and its dependencies, ensure you have [Poetry](https://python-poetry.org/) installed. Then, run the following commands:



Install the dependencies and the project:
   ```bash
   poetry install
   ```

#### Installation with pip
Alternatively, you can install the project's requirements using pip:
```bash
pip install .
```

</details>

---

## 🚀 Quick Start

<details>
<summary>Click to expand</summary>

1. **Load a Scenario**:
   Load a CommonRoad scenario that includes pedestrians using the `CommonRoadFileReader`:
   ```python
   from commonroad.common.file_reader import CommonRoadFileReader

   scenario_path = 'path/to/scenario.xml'
   scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
   ```

2. **Initialize the Simulator**:
   Instantiate the Pedestrian Simulator with the loaded scenario:
   ```python
   from pedestrian_simulator import PedestrianSimulator

   pedestrian_simulator = PedestrianSimulator(scenario)
   ```

3. **Simulate Pedestrian Behavior**:
   Advance the simulation for a specified number of steps:
   ```python
   for timestep in range(100):
       pedestrian_simulator.step_pedestrians(timestep)
   ```

</details>

---

## 🏃 Example Workflow

<details>
<summary>Click to expand</summary>

The repository includes a Minimal Working Example (MWE) in `main.py` to demonstrate the pedestrian simulation workflow:
You can include the pedestrian simulator in any CommonRoad scenario.

</details>

---

## 📈 Visualization

<details>
<summary>Click to expand</summary>

The framework supports detailed visualization of:
- Pedestrian trajectories.
- Interaction forces.
- Vehicle predictions (if vehicles are present in the scenario).

</details>

---

## 🛠 Scenario Creation Tool

<details>
<summary>Click to expand</summary>

The `create_pedestrian_scenario` utility allows you to extend an existing CommonRoad scenario by adding pedestrians, sidewalks, and crosswalks. This tool is highly configurable to adapt to a variety of simulation needs.

### Features
- Add sidewalks and crosswalks to existing scenarios.
- Add pedestrians to the scenario.
- Customize parameters such as pedestrian speed, clustering distance, and position deviations.
- Visualize the generated scenario.

### Example Usage
```python
from pedestrian_simulator.tools.create_pedestrian_scenario import create_new_pedestrian_scenario

input_file = "path/to/scenario.xml"
scenario, planning_problem_set = create_new_pedestrian_scenario(
    input_file=input_file,
    pedestrian_speed=1.2,          # Average pedestrian speed
    pedestrians_per_cluster=2.0,  # Number of pedestrians per cluster
    position_deviation=0.2,       # Deviation in positions within a cluster
    cluster_distance=4.0,         # Distance between pedestrian clusters
    seed=42                       # Random seed for reproducibility
)

# Save or visualize the scenario as needed
```

See also the MWE in `create_pedestrian_scenario.py` for an example workflow.

</details>

---

## 📚 Documentation

<details>
<summary>Click to expand</summary>

For detailed explanations of the attributes and methods, refer to the source code. The key method of the simulator is:
- `step_pedestrians(timestep)`: Advance simulation by one timestep (starting from the given timestep).

</details>

---

# 📇 Contact Info

<details>
<summary>Click to expand</summary>

[Korbinian Moller](mailto:korbinian.moller@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

[Johannes Betz](mailto:johannes.betz@tum.de),
Professorship Autonomous Vehicle Systems,
School of Engineering and Design,
Technical University of Munich,
85748 Garching,
Germany

</details>


---

## 📃 Citation

<details>
<summary>Click to expand</summary>

If you use this Pedestrian Simulator in your research, please cite the related paper:

```bibtex
t.b.d
```

</details>

