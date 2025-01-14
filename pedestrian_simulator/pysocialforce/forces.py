"""Calculate forces for individuals and groups"""
import re
from abc import ABC, abstractmethod

import numpy as np
from numba import njit

from pedestrian_simulator.pysocialforce.potentials import PedPedPotential, PedSpacePotential
from pedestrian_simulator.pysocialforce.fieldofview import FieldOfView
from pedestrian_simulator.pysocialforce.utils import Config, stateutils


def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        super().__init__()
        self.simulator = None
        self.factor = 1.0
        self.config = Config()

    def init(self, simulator, config, delta_t):
        """Load config and scene"""
        # load the sub field corresponding to the force name from global confgi file
        self.config = config.sub_config(camel_to_snake(type(self).__name__))
        if self.config:
            self.factor = self.config("factor", 1.0)

        self.simulator = simulator
        self.delta_t = delta_t

    @abstractmethod
    def _get_force(self) -> np.ndarray:
        """Abstract class to get social forces
            return: an array of force vectors for each pedestrians
        """
        raise NotImplementedError

    def get_force(self, debug=False):
        force = self._get_force()
        return force


class GoalAttractiveForce(Force):
    """accelerate to desired velocity"""

    def _get_force(self):
        tau = self.config("tau")
        preferred_speed = self.config("preferred_speed")
        F0 = (preferred_speed * self.simulator.peds.dir() - self.simulator.peds.vel()) / tau
        return F0 * self.factor


class PedRepulsiveForce(Force):
    """Ped to ped repulsive force"""

    def _get_force(self):
        potential_func = PedPedPotential(delta_t=self.delta_t)
        f_ab = -1.0 * potential_func.grad_r_ab(self.simulator.peds)

        fov = FieldOfView(phi=self.config("fov_phi"), out_of_view_factor=self.config("fov_factor"),)
        w = np.expand_dims(fov(self.simulator.peds.dir(), -f_ab), -1)
        F_ab = w * f_ab
        return np.sum(F_ab, axis=1) * self.factor

class SpaceRepulsiveForce(Force):
    """obstacles to ped repulsive force"""

    def _get_force(self):
        if self.simulator.env.obstacles is None:
            F_aB = np.zeros((self.simulator.peds.size(), 0, 2))
        else:
            potential_func = PedSpacePotential(
                self.simulator.env.obstacles, u0=self.config("u0"), r=self.config("r")
            )
            F_aB = -1.0 * potential_func.grad_r_aB(self.simulator.peds.state)
        return np.sum(F_aB, axis=1) * self.factor

class GroupCoherenceForce(Force):
    """Group coherence force, paper version"""

    def _get_force(self):
        forces = np.zeros((self.simulator.peds.size(), 2))
        if self.simulator.peds.has_group():
            for group in self.simulator.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.simulator.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                vectors, norms = stateutils.normalize(force_vec)
                vectors[norms < threshold] = [0, 0]
                forces[group, :] += vectors
        return forces * self.factor


class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def _get_force(self):
        forces = np.zeros((self.simulator.peds.size(), 2))
        if self.simulator.peds.has_group():
            for group in self.simulator.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.simulator.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def _get_force(self):
        threshold = self.config("threshold", 0.5)
        forces = np.zeros((self.simulator.peds.size(), 2))
        if self.simulator.peds.has_group():
            for group in self.simulator.peds.groups:
                size = len(group)
                member_pos = self.simulator.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.factor


class GroupGazeForce(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.simulator.peds.size(), 2))
        vision_angle = self.config("fov_phi", 100.0)
        directions, _ = stateutils.desired_directions(self.simulator.peds.state)
        if self.simulator.peds.has_group():
            for group in self.simulator.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.simulator.peds.pos()[group, :]
                member_directions = directions[group, :]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, _ = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                com_angles = np.degrees(np.arccos(element_prod))
                rotation = np.radians(
                    [a - vision_angle if a > vision_angle else 0.0 for a in com_angles]
                )
                force = -rotation.reshape(-1, 1) * member_directions
                forces[group, :] += force

        return forces * self.factor


class GroupGazeForceAlt(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.simulator.peds.size(), 2))
        directions, dist = stateutils.desired_directions(self.simulator.peds.state)
        if self.simulator.peds.has_group():
            for group in self.simulator.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.simulator.peds.pos()[group, :]
                member_directions = directions[group, :]
                member_dist = dist[group]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, com_dist = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                force = (
                    com_dist.reshape(-1, 1)
                    * element_prod.reshape(-1, 1)
                    / member_dist.reshape(-1, 1)
                    * member_directions
                )
                forces[group, :] += force

        return forces * self.factor


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def _get_force(self):
        relexation_time = self.config("relaxation_time", 0.5)
        goal_threshold = self.config("goal_threshold", 0.1)
        preferred_speed = self.config("preferred_speed", 1.2)
        pos = self.simulator.peds.pos()
        vel = self.simulator.peds.vel()
        goal = self.simulator.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)
        direction = self.simulator.peds.dir()
        force = np.zeros((self.simulator.peds.size(), 2))
        force[dist > goal_threshold] = (
            direction * preferred_speed - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        force /= relexation_time
        return force * self.factor


class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """

    def _get_force(self):
        lambda_importance = self.config("lambda_importance", 2.0)
        gamma = self.config("gamma", 0.35)
        n = self.config("n", 2)
        n_prime = self.config("n_prime", 3)

        pos_diff = stateutils.each_diff(self.simulator.peds.pos())  # n*(n-1)x2 other - self
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.simulator.peds.vel())  # n*(n-1)x2 self - other

        # compute interaction direction t_ij
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # compute angle theta (between interaction and position difference vector)
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        # compute model parameter B = gamma * ||D||
        B = gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((self.simulator.peds.size(), -1, 2)), axis=1)
        return force * self.factor


class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def _get_force(self):
        sigma = self.config("sigma", 0.2)
        agent_radius = self.config("agent_radius", 0.35)
        threshold = self.config("threshold", 0.2) + agent_radius
        force = np.zeros((self.simulator.peds.size(), 2))
        if len(self.simulator.env.obstacles) == 0:
            return force
        obstacles = np.vstack(self.simulator.env.obstacles)
        pos = self.simulator.peds.pos()

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - agent_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)

        return force * self.factor

class VehicleForce(Force):
    """Calculates the force between this agent and all vehicles in this
    scene.
    :return:  the calculated force
    """

    def _get_force(self):
        
        if len(self.simulator.env.obstacles) == 0:
            return np.zeros((self.simulator.peds.size(), 2))
        obstacles = np.vstack(self.simulator.env.obstacles)
        positions = self.simulator.peds.pos()
        return vehicle_force(obstacles, positions)

@njit
def vehicle_force(obstacles, positions, agent_radius=1.5, sigma=0.2):

    # Compute the difference between every obstacle and every pedestrian
    diff = obstacles[:, np.newaxis, :] - positions[np.newaxis, :, :]

    # Normalize the differences to get directions and distances
    directions, dist = stateutils.normalize(diff)

    # Adjust distances for pedestrian radius
    dist = np.maximum(dist - agent_radius, 0.1)

    # Find the closest obstacle for each pedestrian
    closest_indices = np.argmin(dist, axis=0)  # (n_pedestrians,)

    # Initialize arrays to store the closest distances and directions
    n_pedestrians = positions.shape[0]
    closest_dists = np.zeros(n_pedestrians)
    closest_directions = np.zeros((n_pedestrians, 2))

    # Loop over pedestrians to avoid advanced indexing issues
    for i in range(n_pedestrians):
        closest_dists[i] = dist[closest_indices[i], i]
        closest_directions[i, :] = directions[closest_indices[i], i, :]

    # Compute the forces
    force = -closest_directions / (closest_dists[:, np.newaxis] * sigma) * np.exp(-closest_dists[:, np.newaxis] / sigma)

    return 10 * force
