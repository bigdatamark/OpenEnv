# [MODULE DOC]
# Parameter sets for BehaviorAgent driving styles (speeds in km/h, distances in meters).
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""This module contains the different parameters sets for each behavior."""


class Cautious(object):
    # [CLASS DOC]
    # Cautious style: max_speed 40, larger margins (speed_lim_dist 6, proximity 12, braking 6).
    """Class for Cautious agent."""

    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0


class Normal(object):
    # [CLASS DOC]
    # Normal style: max_speed 50, speed_lim_dist 3, proximity 10, braking distance 5.
    """Class for Normal agent."""

    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0


class Aggressive(object):
    # [CLASS DOC]
    # Aggressive style: max_speed 70, speed_lim_dist 1, proximity 8, braking 4, tailgate_counter -1.
    """Class for Aggressive agent."""

    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 4
    tailgate_counter = -1
