"""This module tracks the state of scene and scene elements like pedestrians, groups and obstacles"""
from typing import Tuple, List
from math import atan2, pi, cos, sin
import numpy as np

from pedestrian_simulator.pysocialforce.utils import stateutils

Line2D = Tuple[float, float, float, float]


class PedState:
    """Tracks the state of pedestrians and social groups"""

    def __init__(self, state, groups, config):
        self.max_speed = config("max_speed", 2.0)

        self.ped_states = []
        self.group_states = []

        self.update(state, groups)

    def update(self, state, groups):
        self.state = state
        self.groups = groups

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        self.ped_states.append(self._state.copy())

    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def policy_id(self) -> np.ndarray:
        return self.state[:, 6].astype(int)
    
    def obstacle_id(self) -> np.ndarray:
        return self.state[:, 7].astype(int)
    
    def dir(self) -> np.ndarray:
        if self.state.shape[1] < 8:
            return self.desired_directions()
        else:
            return self.state[:, 8:10]
    
    def update_dir(self, dir) -> np.ndarray:
        self.state[:, 8:10] = dir

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """Move peds according to forces"""
        # desired velocity
        desired_velocity = self.vel() + 0.1 * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speed)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        # update state
        next_state = self.state
        next_state[:, 0:2] += desired_velocity * 0.1
        next_state[:, 2:4] = desired_velocity
        next_groups = self.groups
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups)

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1


class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacle_lines: List[Line2D]) -> np.ndarray:
        """Input a list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacle_lines is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacle_lines:
                samples = 1+int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)
