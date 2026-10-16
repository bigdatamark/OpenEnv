# [MODULE DOC]
# Vendored CARLA BasicAgent: follows a route from the GlobalRoutePlanner via a LocalPlanner,
# emergency-braking for vehicles ahead and red traffic lights (stop signs ignored). Requires
# the carla and shapely packages.
# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from carla_env.server.carla_agents.navigation.global_route_planner import (
    GlobalRoutePlanner,
)
from carla_env.server.carla_agents.navigation.local_planner import (
    LocalPlanner,
    RoadOption,
)
from carla_env.server.carla_agents.tools.misc import (
    compute_distance,
    get_speed,
    get_trafficlight_trigger_location,
    is_within_distance,
)


class BasicAgent(object):
    # [CLASS DOC]
    # Navigation agent wrapping a vehicle actor. Holds the world/map, a LocalPlanner (PID control
    # along a waypoint queue), a GlobalRoutePlanner, detection thresholds and cached traffic
    # light trigger waypoints.
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(
        self, vehicle, target_speed=20, opt_dict=None, map_inst=None, grp_inst=None
    ):
        # [FUNCTION DOC]
        # Inputs: vehicle actor; target_speed km/h (default 20); opt_dict overrides (ignore_* flags,
        # use_bbs_detection, sampling_resolution, base thresholds, detection_speed_ratio, max_brake,
        # offset; also forwarded to LocalPlanner); optional map_inst / grp_inst to reuse.
        # Invalid map_inst/grp_inst types are ignored with a printed warning and rebuilt.
        # Side effects: mutates opt_dict (sets target_speed); queries world actors for traffic lights.
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

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
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.6
        self._offset = 0

        # Change parameters according to the dictionary
        opt_dict["target_speed"] = target_speed
        if "ignore_traffic_lights" in opt_dict:
            self._ignore_traffic_lights = opt_dict["ignore_traffic_lights"]
        if "ignore_stop_signs" in opt_dict:
            self._ignore_stop_signs = opt_dict["ignore_stop_signs"]
        if "ignore_vehicles" in opt_dict:
            self._ignore_vehicles = opt_dict["ignore_vehicles"]
        if "use_bbs_detection" in opt_dict:
            self._use_bbs_detection = opt_dict["use_bbs_detection"]
        if "sampling_resolution" in opt_dict:
            self._sampling_resolution = opt_dict["sampling_resolution"]
        if "base_tlight_threshold" in opt_dict:
            self._base_tlight_threshold = opt_dict["base_tlight_threshold"]
        if "base_vehicle_threshold" in opt_dict:
            self._base_vehicle_threshold = opt_dict["base_vehicle_threshold"]
        if "detection_speed_ratio" in opt_dict:
            self._speed_ratio = opt_dict["detection_speed_ratio"]
        if "max_brake" in opt_dict:
            self._max_brake = opt_dict["max_brake"]
        if "offset" in opt_dict:
            self._offset = opt_dict["offset"]

        # Initialize the planners
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict=opt_dict, map_inst=self._map
        )
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(
                    self._map, self._sampling_resolution
                )
        else:
            self._global_planner = GlobalRoutePlanner(
                self._map, self._sampling_resolution
            )

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        # [FUNCTION DOC]
        # Sets throttle 0, brake to max_brake and hand_brake False on the control; returns it (steer kept).
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        # [FUNCTION DOC]
        # Sets the agent and local-planner target speed (km/h).
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        # [FUNCTION DOC]
        # Enables/disables following road speed limits in the local planner.
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        # [FUNCTION DOC]
        # Returns the LocalPlanner.
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        # [FUNCTION DOC]
        # Returns the GlobalRoutePlanner.
        # KNOWN ISSUE: docstring says it returns the local planner.
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        # [FUNCTION DOC]
        # Traces a route from start_location (default: the vehicle location, clearing the existing
        # queue) to end_location and loads it into the local planner (queue kept when a start is given).
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._vehicle.get_location()
            clean_queue = True
        else:
            # Use the caller-provided start_location as-is.
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        # [FUNCTION DOC]
        # Passes a list of (waypoint, RoadOption) to the local planner with the given flags.
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan, stop_waypoint_creation=stop_waypoint_creation, clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        # [FUNCTION DOC]
        # Returns the global planner route between two waypoints' locations as (waypoint, RoadOption) pairs.
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        # [FUNCTION DOC]
        # Runs one control step: checks vehicles ahead and red lights within thresholds scaled by speed
        # (m/s * detection ratio), gets the local planner control, and applies an emergency stop if a
        # hazard was detected. Returns a carla.VehicleControl.
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = (
            self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        )
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(
            vehicle_list, max_vehicle_distance
        )
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = (
            self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        )
        affected_by_tlight, _ = self._affected_by_traffic_light(
            self._lights_list, max_tlight_distance
        )
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

    def done(self):
        # [FUNCTION DOC]
        # True when the local planner's route is finished.
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        # [FUNCTION DOC]
        # Sets whether traffic lights are ignored.
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        # [FUNCTION DOC]
        # Sets whether stop signs are ignored (stored only; this agent never checks stop signs).
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        # [FUNCTION DOC]
        # Sets whether vehicles are ignored.
        # KNOWN ISSUE: docstring mentions stop signs instead of vehicles.
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def set_offset(self, offset):
        # [FUNCTION DOC]
        # Sets the lateral offset in the local planner.
        """Sets an offset for the vehicle"""
        self._local_planner.set_offset(offset)

    def lane_change(
        self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2
    ):
        # [FUNCTION DOC]
        # Builds a lane-change path from the current waypoint using distances = given times x current
        # speed (m/s) without lane-marking checks, and sets it as the global plan.
        # KNOWN ISSUE: when no path is found it prints a warning but still sets the empty plan.
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution,
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        # [FUNCTION DOC]
        # HELPER: Returns (True, light) if a red light whose trigger waypoint is on the ego road, faces the
        # same direction and is within max_distance ahead (0-90 deg cone) affects the ego; remembers it
        # until it turns non-red. Returns (False, None) otherwise or when lights are ignored. Defaults to
        # all scene lights and the base threshold.
        # Side effects: caches trigger waypoints in _lights_map, updates _last_traffic_light.
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if (
                trigger_wp.transform.location.distance(ego_vehicle_location)
                > max_distance
            ):
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(
                trigger_wp.transform,
                self._vehicle.get_transform(),
                max_distance,
                [0, 90],
            ):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(
        self,
        vehicle_list=None,
        max_distance=None,
        up_angle_th=90,
        low_angle_th=0,
        lane_offset=0,
    ):
        # [FUNCTION DOC]
        # HELPER: Looks for a vehicle blocking the path within max_distance. Uses polygon intersection of
        # the route corridor with target bounding boxes when bbs detection is on, the ego invades the
        # opposite lane, or in junctions; otherwise checks lane/road match (current or 3 steps ahead,
        # with lane_offset) and whether the target's rear is within the distance/angle cone of the ego
        # front. Returns (detected, vehicle or None, distance or -1).
        # KNOWN ISSUE: the ego front transform and target rear transform are computed by mutating aliased
        # transform objects, so the "original" transforms are shifted too.
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """

        def get_route_polygon():
            # [FUNCTION DOC]
            # HELPER: Builds a shapely Polygon from points offset left/right of the ego and of upcoming plan
            # waypoints within max_distance; returns None if fewer than 3 points.
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(
                    r_ext * r_vec.x, r_ext * r_vec.y
                )
                p2 = wp.transform.location + carla.Location(
                    l_ext * r_vec.x, l_ext * r_vec.y
                )
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector()
        )

        opposite_invasion = (
            abs(self._offset) + self._vehicle.bounding_box.extent.y
            > ego_wpt.lane_width / 2
        )
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(
                target_transform.location, lane_type=carla.LaneType.Any
            )

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:
                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(
                    target_vehicle.get_transform()
                )
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (
                        True,
                        target_vehicle,
                        compute_distance(target_vehicle.get_location(), ego_location),
                    )

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:
                if (
                    target_wpt.road_id != ego_wpt.road_id
                    or target_wpt.lane_id != ego_wpt.lane_id + lane_offset
                ):
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(
                        steps=3
                    )[0]
                    if not next_wpt:
                        continue
                    if (
                        target_wpt.road_id != next_wpt.road_id
                        or target_wpt.lane_id != next_wpt.lane_id + lane_offset
                    ):
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(
                    target_rear_transform,
                    ego_front_transform,
                    max_distance,
                    [low_angle_th, up_angle_th],
                ):
                    return (
                        True,
                        target_vehicle,
                        compute_distance(
                            target_transform.location, ego_transform.location
                        ),
                    )

        return (False, None, -1)

    def _generate_lane_change_path(
        self,
        waypoint,
        direction="left",
        distance_same_lane=10,
        distance_other_lane=25,
        lane_change_distance=25,
        check=True,
        lane_changes=1,
        step_distance=2,
    ):
        # [FUNCTION DOC]
        # HELPER: Returns a list of (waypoint, RoadOption): distance_same_lane of lane following, then
        # lane_changes side-lane steps (optionally checking lane-change markings) in direction, then
        # distance_other_lane of following. Returns an empty list if direction is invalid or any
        # waypoint/side driving lane is unavailable.
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == "left":
            option = RoadOption.CHANGELANELEFT
        elif direction == "right":
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:
            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == "left":
                if check and str(next_wp.lane_change) not in ["Left", "Both"]:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ["Right", "Both"]:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location
            )
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
