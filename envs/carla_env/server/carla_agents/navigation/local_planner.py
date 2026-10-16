# [MODULE DOC]
# Vendored CARLA LocalPlanner: follows a queue of (waypoint, RoadOption) pairs with a
# VehiclePIDController, generating random waypoints ahead when no global plan stops it. Also
# defines the RoadOption enum and helpers classifying branch directions.
# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This module contains a local planner to perform low-level waypoint following based on PID controllers."""

import random
from collections import deque
from enum import IntEnum

import carla

from carla_env.server.carla_agents.navigation.controller import VehiclePIDController
from carla_env.server.carla_agents.tools.misc import draw_waypoints, get_speed


class RoadOption(IntEnum):
    # [CLASS DOC]
    # Integer enum of topological moves: VOID, LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT,
    # CHANGELANERIGHT.
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """

    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    # [CLASS DOC]
    # Waypoint follower. Holds a bounded deque (10000) of planned waypoints, the current target
    # waypoint/option, PID parameters, speed (km/h) and the speed-limit following flag.
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice,
    unless a given global plan has already been specified.
    """

    def __init__(self, vehicle, opt_dict=None, map_inst=None):
        # [FUNCTION DOC]
        # Configures defaults (dt 0.05, 20 km/h, 2 m sampling, PID gains, throttle/brake/steer limits,
        # min-distance parameters) overridden by opt_dict keys, resolves the map (invalid map_inst ignored
        # with a printed warning) and initializes the controller and first waypoint.
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with different parameters:
            dt: time between simulation steps
            target_speed: desired cruise speed in Km/h
            sampling_radius: distance between the waypoints part of the plan
            lateral_control_dict: values of the lateral PID controller
            longitudinal_control_dict: values of the longitudinal PID controller
            max_throttle: maximum throttle applied to the vehicle
            max_brake: maximum brake applied to the vehicle
            max_steering: maximum steering applied to the vehicle
            offset: distance between the route waypoints and the center of the lane
        :param map_inst: carla.Map instance to avoid the expensive call of getting it.
        """
        if opt_dict is None:
            opt_dict = {}
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        self._vehicle_controller = None
        self.target_waypoint = None
        self.target_road_option = None

        self._waypoints_queue = deque(maxlen=10000)
        self._min_waypoint_queue_length = 100
        self._stop_waypoint_creation = False

        # Base parameters
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0  # Km/h
        self._sampling_radius = 2.0
        self._args_lateral_dict = {"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": self._dt}
        self._args_longitudinal_dict = {
            "K_P": 1.5,
            "K_I": 0.05,
            "K_D": 0.2,
            "dt": self._dt,
        }
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0
        self._base_min_distance = 3.0
        self._distance_ratio = 0.5
        self._follow_speed_limits = False

        # Overload parameters
        if opt_dict:
            if "dt" in opt_dict:
                self._dt = opt_dict["dt"]
            if "target_speed" in opt_dict:
                self._target_speed = opt_dict["target_speed"]
            if "sampling_radius" in opt_dict:
                self._sampling_radius = opt_dict["sampling_radius"]
            if "lateral_control_dict" in opt_dict:
                self._args_lateral_dict = opt_dict["lateral_control_dict"]
            if "longitudinal_control_dict" in opt_dict:
                self._args_longitudinal_dict = opt_dict["longitudinal_control_dict"]
            if "max_throttle" in opt_dict:
                self._max_throt = opt_dict["max_throttle"]
            if "max_brake" in opt_dict:
                self._max_brake = opt_dict["max_brake"]
            if "max_steering" in opt_dict:
                self._max_steer = opt_dict["max_steering"]
            if "offset" in opt_dict:
                self._offset = opt_dict["offset"]
            if "base_min_distance" in opt_dict:
                self._base_min_distance = opt_dict["base_min_distance"]
            if "distance_ratio" in opt_dict:
                self._distance_ratio = opt_dict["distance_ratio"]
            if "follow_speed_limits" in opt_dict:
                self._follow_speed_limits = opt_dict["follow_speed_limits"]

        # initializing controller
        self._init_controller()

    def reset_vehicle(self):
        # [FUNCTION DOC]
        # Clears the vehicle reference (sets it to None).
        """Reset the ego-vehicle"""
        self._vehicle = None

    def _init_controller(self):
        # [FUNCTION DOC]
        # HELPER: Builds the VehiclePIDController and seeds the queue with the vehicle's current waypoint
        # as LANEFOLLOW.
        """Controller initialization"""
        self._vehicle_controller = VehiclePIDController(
            self._vehicle,
            args_lateral=self._args_lateral_dict,
            args_longitudinal=self._args_longitudinal_dict,
            offset=self._offset,
            max_throttle=self._max_throt,
            max_brake=self._max_brake,
            max_steering=self._max_steer,
        )

        # Compute the current vehicle waypoint
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (
            current_waypoint,
            RoadOption.LANEFOLLOW,
        )
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed):
        # [FUNCTION DOC]
        # Sets the target speed (km/h); prints a warning if speed-limit following is active (which will
        # override it in run_step).
        """
        Changes the target speed

        :param speed: new target speed in Km/h
        :return:
        """
        if self._follow_speed_limits:
            print(
                "WARNING: The max speed is currently set to follow the speed limits. "
                "Use 'follow_speed_limits' to deactivate this"
            )
        self._target_speed = speed

    def follow_speed_limits(self, value=True):
        # [FUNCTION DOC]
        # Enables/disables using the vehicle speed limit as target speed.
        """
        Activates a flag that makes the max speed dynamically vary according to the spped limits

        :param value: bool
        :return:
        """
        self._follow_speed_limits = value

    def _compute_next_waypoints(self, k=1):
        # [FUNCTION DOC]
        # HELPER: Appends up to k waypoints (bounded by free queue space) by stepping sampling_radius from
        # the last one; at forks picks a random branch option. Stops early at dead ends.
        # KNOWN ISSUE: when several branches share the same RoadOption, only the first such branch can ever
        # be chosen.
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(
        self, current_plan, stop_waypoint_creation=True, clean_queue=True
    ):
        # [FUNCTION DOC]
        # Optionally clears the queue, grows the deque if the combined plan exceeds its capacity, appends
        # the plan and sets whether random waypoint creation stops.
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs
        The 'clean_queue` parameter erases the previous plan if True, otherwise, it adds it to the old one
        The 'stop_waypoint_creation' flag stops the automatic creation of random waypoints

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :param stop_waypoint_creation: bool
        :param clean_queue: bool
        :return:
        """
        if clean_queue:
            self._waypoints_queue.clear()

        # Remake the waypoints queue if the new plan has a higher length than the queue
        new_plan_length = len(current_plan) + len(self._waypoints_queue)
        if new_plan_length > self._waypoints_queue.maxlen:
            new_waypoint_queue = deque(maxlen=new_plan_length)
            for wp in self._waypoints_queue:
                new_waypoint_queue.append(wp)
            self._waypoints_queue = new_waypoint_queue

        for elem in current_plan:
            self._waypoints_queue.append(elem)

        self._stop_waypoint_creation = stop_waypoint_creation

    def set_offset(self, offset):
        # [FUNCTION DOC]
        # Sets the lateral offset on the PID controller.
        """Sets an offset for the vehicle"""
        self._vehicle_controller.set_offset(offset)

    def run_step(self, debug=False):
        # [FUNCTION DOC]
        # Optionally adopts the speed limit, tops up random waypoints when allowed, pops waypoints closer
        # than base_min_distance + ratio * speed (m/s) (the last only within 1 m), then returns PID control
        # toward the first queued waypoint, or full brake if the queue is empty. With debug, draws the
        # target waypoint. Returns a carla.VehicleControl.
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()

        # Add more waypoints too few in the horizon
        if (
            not self._stop_waypoint_creation
            and len(self._waypoints_queue) < self._min_waypoint_queue_length
        ):
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        self._min_distance = (
            self._base_min_distance + self._distance_ratio * vehicle_speed
        )

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(
                self._target_speed, self.target_waypoint
            )

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        return control

    def get_incoming_waypoint_and_direction(self, steps=3):
        # [FUNCTION DOC]
        # Returns the (waypoint, option) steps ahead in the queue, the last entry if fewer, or
        # (None, VOID) if the queue is empty.
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]

        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError as i:
                return None, RoadOption.VOID

    def get_plan(self):
        # [FUNCTION DOC]
        # Returns the live waypoint queue.
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self):
        # [FUNCTION DOC]
        # True when the waypoint queue is empty.
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0


def _retrieve_options(list_waypoints, current_waypoint):
    # [FUNCTION DOC]
    # HELPER: For each candidate waypoint, classifies the connection from the current waypoint to the
    # point 3 m beyond the candidate. Returns the list of RoadOptions.
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    # [FUNCTION DOC]
    # HELPER: Compares yaw angles modulo 180: within threshold (35 deg) of 0/180 -> STRAIGHT, above 90
    # -> LEFT, else RIGHT.
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
