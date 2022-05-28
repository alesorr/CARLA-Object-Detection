#!/usr/bin/env python3
import numpy as np
import math



# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._trafficlights_fences          = []
        self._pedestrians_fences            = []
        self._vehicles_fences               = []
        self._measurement_non_player_agents = []
        self._trafficlights_visited         = []
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._traffic_light_id              = None
        self._why_decelerate                = 0 # 1: traffic light found; 2: pedestrian found; 3: vehicle found.
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead
    
    def set_measurement_non_player_agents(self, measurement_non_player_agents):
        self._measurement_non_player_agents = measurement_non_player_agents
    
    def set_trafficlights_fences(self, trafficlights_fences):
        self._trafficlights_fences = trafficlights_fences
        self._trafficlights_visited = [False]*len(self._trafficlights_fences)
    
    def set_pedestrians_fences(self, pedestrians_fences):
        self._pedestrians_fences = pedestrians_fences
    
    def set_vehicles_fences(self, vehicles_fences):
        self._vehicles_fences = vehicles_fences

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any traffic light lines, pedestrian lines or vehicle lines.
        # If it does, then ensure that the goal state enforces the car to
        # be stopped before the relative line. You should use the
        # get_closest_index(), get_goal_index(), and check_for_traffic_lights(),
        # check_for_pedestrians(), check_for_vehicles() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_traffic_lights(), check_for_pedestrians(),
        # check_for_vehicles() function to understand it.
        if self._state == FOLLOW_LANE:
            #print("FOLLOW_LANE")
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            # So, check the index set between closest_index and goal_index
            # for traffic lights, and compute the goal state accordingly.
            goal_index, traffic_light_found, traffic_light_id = self.check_for_traffic_lights(waypoints, closest_index, goal_index)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # Moreover, check the index set between closest_index and goal_index
            # for pedestrians, and compute the goal state accordingly.
            goal_index, pedestrians_found = self.check_for_pedestrians(waypoints, closest_index, goal_index)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            # Finally, check the index set between closest_index and goal_index
            # for vehicles, and compute the goal state accordingly.
            goal_index, vehicles_found = self.check_for_vehicles(waypoints, closest_index, goal_index)
            self._goal_index = goal_index
            self._goal_state = waypoints[goal_index]

            is_yellow_or_red = False
            for agent in self._measurement_non_player_agents:
                if agent.id == traffic_light_id:
                    # Checking if traffic light is yellow or red
                    if agent.traffic_light.state == 1 or agent.traffic_light.state == 2:
                        self._traffic_light_id = traffic_light_id
                        is_yellow_or_red = True
                        break

            # If traffic light found, set the goal to zero speed, then transition to 
            # the deceleration state.
            if traffic_light_found and is_yellow_or_red:
                self._goal_state[2] = 0
                self._why_decelerate = 1
                self._state = DECELERATE_TO_STOP 

            # If pedestrian found, set the goal to zero speed, then transition to 
            # the deceleration state.
            if pedestrians_found:
                self._goal_state[2] = 0
                self._why_decelerate = 2
                self._state = DECELERATE_TO_STOP
            
            # If vehicle found, set the goal to zero speed, then transition to 
            # the deceleration state.
            if vehicles_found:
                self._goal_state[2] = 0
                self._why_decelerate = 3
                self._state = DECELERATE_TO_STOP 

        # In this state, check if we have reached a red traffic light,
        # a pedestrian or a vehicle as obstacles. Use the# closed loop
        # speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD. If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            #print("DECELERATE_TO_STOP")
            if abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
            
            # Check on traffic lights
            if self._why_decelerate == 1:
                #print("TRAFFIC LIGHT")
                for agent in self._measurement_non_player_agents:
                    if agent.id == self._traffic_light_id:
                        if agent.traffic_light.state == 0:
                            self._traffic_light_id = None
                            self._why_decelerate = 0
                            self._state = FOLLOW_LANE
                            break
            # Check on pedestrians
            elif self._why_decelerate == 2:
                #print("PEDESTRIANS")
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1
                goal_index, pedestrians_found = self.check_for_pedestrians(waypoints, closest_index, goal_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                if not pedestrians_found:
                    self._why_decelerate = 0
                    self._state = FOLLOW_LANE
            # Check on vehicles
            elif self._why_decelerate == 3:
                #print("VEHICLES")
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1
                goal_index, vehicles_found = self.check_for_vehicles(waypoints, closest_index, goal_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                if not vehicles_found:
                    self._why_decelerate = 0
                    self._state = FOLLOW_LANE

        # In this state, check to see until we have to stay stopped.
        # In case of traffic light, we have to stay stopped until
        # green light is on; in case of pedestrians, we have to stay
        # stopped until they are obstructing our lane; in case of vehicles,
        # we have to stay stopped until they are obstructing our lane.
        elif self._state == STAY_STOPPED:
            #print("STAY STOPPED")
            # Check on traffic lights
            if self._why_decelerate == 1:
                #print("TRAFFIC LIGHT")
                for agent in self._measurement_non_player_agents:
                    if agent.id == self._traffic_light_id:
                        if agent.traffic_light.state == 0:
                            self._traffic_light_id = None
                            self._why_decelerate = 0
                            self._state = FOLLOW_LANE
                            break
            # Check on pedestrians
            elif self._why_decelerate == 2:
                #print("PEDESTRIANS")
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1
                goal_index, pedestrians_found = self.check_for_pedestrians(waypoints, closest_index, goal_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                if not pedestrians_found:
                    self._why_decelerate = 0
                    self._state = FOLLOW_LANE
            # Check on vehicles
            elif self._why_decelerate == 3:
                #print("VEHICLES")
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1
                goal_index, vehicles_found = self.check_for_vehicles(waypoints, closest_index, goal_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]
                if not vehicles_found:
                    self._why_decelerate = 0
                    self._state = FOLLOW_LANE

        else:
            raise ValueError('Invalid state value.')

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    # Checks the given segment of the waypoint list to see if it
    # intersects with a traffic light line. If any index does, return the
    # new goal state accordingly.
    def check_for_traffic_lights(self, waypoints, closest_index, goal_index):
        """Checks for a traffic light that is intervening the goal path.

        Checks for a traffic light that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a traffic light line), and a
        boolean flag indicating if a traffic light obstruction was found.
        
        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), trafficlight_found, traffic_light_id]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                trafficlight_found: Boolean flag for whether a traffic_light was found or not
                trafficlight_id: useful to analyze its state over time
        """
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the traffic light lines.
            intersect_flag = False
            for key,trafficlights_fence in enumerate(self._trafficlights_fences):
                if self._trafficlights_visited[key]: continue

                wp_1   = np.array(waypoints[i][0:2])
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(trafficlights_fence[0:2])
                s_2    = np.array(trafficlights_fence[2:4])

                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a traffic light line, update
                # the goal state to traffic before the goal line.
                if intersect_flag:
                    goal_index = i
                    self._trafficlights_visited[key] = True
                    return goal_index, True, self._trafficlights_fences[key][4] # traffic light id

        return goal_index, False, -1

    # Checks the given segment of the waypoint list to see if it
    # intersects with a pedestrian line. If any index does, return the
    # new goal state accordingly.
    def check_for_pedestrians(self, waypoints, closest_index, goal_index):
        """Checks for a pedestrian that is intervening the goal path.

        Checks for a pedestrian that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a pedestrian line), and a
        boolean flag indicating if a pedestrian obstruction was found.
        
        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), pedestrian_found]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                pedestrian_found: Boolean flag for whether a pedestrian was found or not
        """
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the traffic lines.
            intersect_flag = False
            for key,pedestrians_fence in enumerate(self._pedestrians_fences):

                wp_1   = np.array(waypoints[i][0:2])
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(pedestrians_fence[0:2])
                s_2    = np.array(pedestrians_fence[2:4])

                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a pedestrian line, update
                # the goal state to traffic before the goal line.
                if intersect_flag:
                    goal_index = i
                    return goal_index, True

        return goal_index, False
    
    # Checks the given segment of the waypoint list to see if it
    # intersects with a vehicle line. If any index does, return the
    # new goal state accordingly.
    def check_for_vehicles(self, waypoints, closest_index, goal_index):
        """Checks for a vehicle that is intervening the goal path.

        Checks for a vehicle that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a vehicle line), and a
        boolean flag indicating if a vehicle obstruction was found.
        
        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
                goal_index (current): Current goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
        variables to set:
            [goal_index (updated), vehicle_found]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                vehicle_found: Boolean flag for whether a vehicle was found or not
        """
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the traffic lines.
            intersect_flag = False
            for key,vehicles_fence in enumerate(self._vehicles_fences):

                wp_1   = np.array(waypoints[i][0:2])
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(vehicles_fence[0:2])
                s_2    = np.array(vehicles_fence[2:4])

                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a vehicle line, update
                # the goal state to traffic before the goal line.
                if intersect_flag:
                    goal_index = i
                    return goal_index, True

        return goal_index, False
    
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.   
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), 
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector, 
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], 
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
