import sys
from collections import defaultdict, deque
from functools import cached_property
from typing import List, Dict, Tuple, Optional, Deque, Generator, Type, Union

from mnms.demand import User
from mnms.log import create_logger
from mnms.mobility_service.abstract import AbstractMobilityService
from mnms.time import Dt, Time
from mnms.tools.cost import create_service_costs
from mnms.tools.exceptions import VehicleNotFoundError
from mnms.vehicles.veh_type import VehicleActivityServing, Vehicle, VehicleActivityStop, VehicleActivityRepositioning, \
    ActivityType, VehicleActivityPickup, VehicleActivity

log = create_logger(__name__)


def _insert_in_activity(pu_node, ind_pu, do_node, ind_do, user, veh):
    if veh.activity is not None and veh.activity.activity_type is not ActivityType.STOP:
        activities_including_curr = [veh.activity] + [a for a in veh.activities]
        decrement_insert_index = True
    elif veh.activity is not None and veh.activity.activity_type == ActivityType.STOP:
        activities_including_curr = [a for a in veh.activities] + [veh.activity]
        decrement_insert_index = False
    else:
        activities_including_curr = [a for a in veh.activities]
        decrement_insert_index = False
    if ind_pu == ind_do:
        # Insertion modifies only one activity in vehicles' activities
        ind = ind_pu
        activity_to_modify = activities_including_curr[ind]
        pu_ind_inpath = veh.path_to_nodes(activity_to_modify.path).index(pu_node)
        do_ind_inpath = veh.path_to_nodes(activity_to_modify.path).index(do_node)
        if ind == 0:
            # activity_to_modify have begun, pickup activity path should start
            # at vehicle current node
            start_ind_inpath = veh.path_to_nodes(activity_to_modify.path).index(veh._current_node)
        else:
            # activity_to_modify has not begun, pickup activity path should start
            # at the beginning of activity_to_modify path
            start_ind_inpath = 0
        # Deduce pickup and serving activities
        pu_path = activity_to_modify.path[start_ind_inpath:pu_ind_inpath]
        do_path = activity_to_modify.path[pu_ind_inpath:do_ind_inpath]
        pu_activity = VehicleActivityPickup(node=pu_node,
                                            path=pu_path,
                                            user=user)
        do_activity = VehicleActivityServing(node=do_node,
                                            path=do_path,
                                            user=user)
        # Modify activity_to_modify
        activity_to_modify.modify_path(activity_to_modify.path[do_ind_inpath:])
        # Insert the new activities and the modified one
        if ind == 0 and veh.activity is not None and veh.activity.activity_type is not ActivityType.STOP:
            # Interrupt current activity and insert the new activities plus the
            # modified one
            veh.activity = None
            # Modify length to travel on the first link of pu_activity.path with
            # current remaining_link_length to prevent restarting the travel of
            # this link
            pu_activity.path[0] = (pu_activity.path[0][0], veh._remaining_link_length)
            pu_activity.reset_path_iterator()
            for a in reversed([pu_activity, do_activity, activity_to_modify]):
                if decrement_insert_index:
                    veh.activities.insert(max(0, ind-1), a)
                else:
                    veh.activities.insert(ind, a)
        else:
            # Only insert the new activities, path of activity_to_modify has
            # already been modified
            for a in reversed([pu_activity, do_activity]):
                if decrement_insert_index:
                    veh.activities.insert(max(0, ind-1), a)
                else:
                    veh.activities.insert(ind, a)
    else:
        assert ind_pu < ind_do, "Index where pickup activity is going to be inserted in "\
            "vehicle's activities is greater than index where serving activity is going "\
            "to be inserted: this is not consistent."
        # Start by inserting serving activity since it is located after pickup
        activity_to_modify_do = activities_including_curr[ind_do]
        do_ind_inpath = veh.path_to_nodes(activity_to_modify_do.path).index(do_node)
        do_path = activity_to_modify_do.path[:do_ind_inpath]
        do_activity = VehicleActivityServing(node=do_node,
                                            path=do_path,
                                            user=user)
        activity_to_modify_do.modify_path(activity_to_modify_do.path[do_ind_inpath:])
        if decrement_insert_index:
            veh.activities.insert(max(0, ind_do-1), do_activity)
        else:
            veh.activities.insert(ind_do, do_activity)
        # Then insert pickup activity
        activity_to_modify_pu = activities_including_curr[ind_pu]
        pu_ind_inpath = veh.path_to_nodes(activity_to_modify_pu.path).index(pu_node)
        if ind_pu == 0:
            start_ind_inpath = veh.path_to_nodes(activity_to_modify_pu.path).index(veh._current_node)
        else:
            start_ind_inpath = 0
        pu_path = activity_to_modify_pu.path[start_ind_inpath:pu_ind_inpath]
        pu_activity = VehicleActivityPickup(node=pu_node,
                                            path=pu_path,
                                            user=user)
        activity_to_modify_pu.modify_path(activity_to_modify_pu.path[pu_ind_inpath:])
        if ind_pu == 0 and veh.activity is not None and veh.activity.activity_type is not ActivityType.STOP:
            # Interrupt current activity and insert the pickup activity plus the
            # modified one
            veh.activity = None
            # Modify length to travel on the first link of pu_activity.path with
            # current remaining_link_length to prevent restarting the travel of
            # this link
            if pu_activity.path:
                pu_activity.path[0] = (pu_activity.path[0][0], veh._remaining_link_length)
                pu_activity.reset_path_iterator()
            for a in reversed([pu_activity, activity_to_modify_pu]):
                if decrement_insert_index:
                    veh.activities.insert(max(0, ind_pu-1), a)
                else:
                    veh.activities.insert(ind_pu, a)
        else:
            # Only insert the pickup activity, path of activity_to_modify_pu has
            # already been modified
            if decrement_insert_index:
                veh.activities.insert(max(0, ind_pu-1), pu_activity)
            else:
                veh.activities.insert(ind_pu, pu_activity)


class PublicTransportMobilityService(AbstractMobilityService):
    def __init__(self, _id: str, veh_capacity=50):
        """
        Implement a public transport mobility service, it can create lines

        Args:
            _id: The id of the service
            veh_capacity: The capacity of the vehicle using this service
        """
        super(PublicTransportMobilityService, self).__init__(_id, veh_capacity=veh_capacity, dt_matching=0,
                                                             dt_periodic_maintenance=0)
        self.vehicles: Dict[str, Deque[Vehicle]] = defaultdict(deque)
        self._timetable_iter: Dict[str, Generator[Time, None, None]] = dict()
        self._current_time_table: Dict[str, Time] = dict()
        self._next_time_table: Dict[str, Time] = dict()
        self._next_veh_departure: Dict[str, Optional[Tuple[Time, Vehicle]]] = defaultdict(lambda: None)

        self.gnodes = None

    @cached_property
    def lines(self):
        return self.layer.lines

    def clean_arrived_vehicles(self, lid: str):
        if len(self.vehicles[lid]) > 0:
            first_veh = self.vehicles[lid][-1]
            if first_veh.activity_type is VehicleActivityStop:
                log.info(f"Deleting arrived veh: {first_veh}")
                self.vehicles[lid].pop()
                self.fleet.delete_vehicle(first_veh.id)
                self.clean_arrived_vehicles(lid)

    def construct_public_transport_path(self, lid):
        veh_path = list()
        path = self.lines[lid]['nodes']
        for i in range(len(path) - 1):
            unode = path[i]
            dnode = path[i + 1]
            key = (unode, dnode)
            link = self.graph.nodes[unode].adj[dnode]
            veh_path.append((key, link.length))
        return veh_path

    def new_departures(self, time, dt, lid: str, all_departures=None):
        """
        Recursive function returning all the departures of a public transport line during the current time step

        Args:
            time: The current time
            dt: The time step
            all_departures:
        """
        veh_path = self.construct_public_transport_path(lid)
        end_node = self.lines[lid]['nodes'][-1]
        start_node = self.lines[lid]['nodes'][0]

        if all_departures is None:
            if self._next_veh_departure[lid] is None:
                new_veh = self.fleet.create_vehicle(start_node,
                                                    capacity=self._veh_capacity,
                                                    activities=[VehicleActivityStop(node=end_node,
                                                                                    path=veh_path)])
                new_veh._current_link = veh_path[0][0]
                new_veh._remaining_link_length = veh_path[0][1]
                self._next_veh_departure[lid] = (self._current_time_table[lid], new_veh)
                log.info(f"Next departure {new_veh}")
            all_departures = list()

        if time > self._current_time_table[lid]:
            self._current_time_table[lid] = self._next_time_table[lid]
            try:
                self._next_time_table[lid] = next(self._timetable_iter[lid])
            except StopIteration:
                return all_departures
            self.new_departures(time, dt, lid, all_departures)

        next_time = time.add_time(dt)
        if time <= self._current_time_table[lid] < next_time:
            log.info(f"New departure {self._next_veh_departure[lid][1]}")
            start_veh = self._next_veh_departure[lid][1]
            stop_activity = start_veh.activity
            repo_activity = VehicleActivityRepositioning(stop_activity.node,
                                                         stop_activity.path,
                                                         stop_activity.user)
            start_veh.add_activities([repo_activity])
            start_veh.next_activity(time)

            all_departures.append(start_veh)
            self.vehicles[lid].appendleft(self._next_veh_departure[lid][1])
            self._current_time_table[lid] = self._next_time_table[lid]
            try:
                self._next_time_table[lid] = next(self._timetable_iter[lid])

                new_veh = self.fleet.create_vehicle(start_node,
                                                    capacity=self._veh_capacity,
                                                    activities=[VehicleActivityStop(node=end_node,
                                                                                    path=veh_path)])
                new_veh._current_link = veh_path[0][0]
                new_veh._remaining_link_length = veh_path[0][1]
                self._next_veh_departure[lid] = (self._next_time_table[lid], new_veh)
                log.info(f"Next departure {new_veh}")
            except StopIteration:
                self._next_veh_departure[lid] = None
                return all_departures
            self.new_departures(time, dt, lid, all_departures)

        return all_departures

    def add_passenger(self, user: User, drop_node: str, veh: Vehicle, line_nodes: List[str]):

        log.info(f"Add passenger {user} -> {veh}")
        user.set_state_waiting_vehicle(veh)

        pu_node_ind = line_nodes.index(user._current_node)
        do_node_ind = line_nodes.index(drop_node)

        assert pu_node_ind <= do_node_ind, f'Pickup index {pu_node_ind} should necessarily take place '\
            f'before dropoff index {do_node_ind} on the public transport line for User {user.id}.'

        # Get the indexes of veh.activities where pickup and serving activities
        # should be inserted
        if veh.activity is not None and veh.activity.activity_type is not ActivityType.STOP:
            activities_including_curr = [veh.activity] + [a for a in veh.activities]
        elif veh.activity is not None and veh.activity.activity_type == ActivityType.STOP:
            activities_including_curr = [a for a in veh.activities] + [veh.activity]
        else:
            activities_including_curr = [a for a in veh.activities]
        ind_pu = -1
        for ind, activity in enumerate(activities_including_curr):
            activity_node = activity.node
            activity_node_ind = line_nodes.index(activity_node)
            if pu_node_ind <= activity_node_ind and ind_pu == -1:
                ind_pu = ind
            if do_node_ind <= activity_node_ind:
                ind_do = ind
                break # if we found dropoff we necessarily have found pickup before

        # Insert the activities corresponding to pickup and serving in vehicles' activities
        _insert_in_activity(user._current_node, ind_pu, drop_node, ind_do, user, veh)

    def estimation_pickup_time(self, user: User, veh: Vehicle, line: dict):
        user_node = user._current_node
        veh_link_borders = veh.current_link
        veh_link_length = self.gnodes[veh_link_borders[0]].adj[veh_link_borders[1]].length
        veh_remaining_length = veh.remaining_link_length
        veh_traveled_dist_link = veh_link_length - veh_remaining_length

        line_stops = line["nodes"]
        ind_user = line_stops.index(user_node)
        ind_veh = line_stops.index(veh_link_borders[0])

        path = line_stops[ind_veh:ind_user+1]
        dist = 0
        for i in range(len(path)-1):
            dist += self.gnodes[path[i]].adj[path[i+1]].length
        dist -= veh_traveled_dist_link
        # NB: if veh has not been moved yet (stopped at the first station of the
        #     line, speed of veh corresponds to the initial speed, it may be different
        #     from the speed computed in the MFD flow so estimation of pickup time
        #     is incorrect)

        dt = Dt(seconds=dist/veh.speed)

        return dt

    def request(self, user: User, drop_node: str) -> Dt:
        # for user, drop_node in users.values():
        start = user._current_node

        chosen_veh = None
        chosen_line = None

        # Select the proper line for user
        for lid, line in self.lines.items():
            if start in line['nodes']:
                chosen_line = line
                user_line_id = lid
                break
        else:
            log.error(f'{user} start is not in the PublicTransport mobility service {self.id}')
            sys.exit(-1)

        if not self.gnodes[start].radj:
            departure_time, waiting_veh = self._next_veh_departure[user_line_id]
            chosen_veh = waiting_veh
        else:
            ind_start = chosen_line["nodes"].index(start)
            for veh in reversed(list(self.vehicles[user_line_id])):
                ind_curr_veh = chosen_line["nodes"].index(veh.current_link[1])
                if ind_curr_veh <= ind_start:
                    chosen_veh = veh
                    break
            else:
                departure_time, waiting_veh = self._next_veh_departure[user_line_id]
                chosen_veh = waiting_veh

        self._cache_request_vehicles[user.id] = (chosen_veh, chosen_line)

        return self.estimation_pickup_time(user, chosen_veh, chosen_line)

    def matching(self, user: User, drop_node: str):
        veh, line = self._cache_request_vehicles[user.id]
        log.info(f'User {user.id} matched with vehicle {veh.id} of mobility service {self._id}')
        self.add_passenger(user, drop_node, veh, line["nodes"])

    def step_maintenance(self, dt: Dt):
        self.gnodes = self.graph.nodes
        for lid in self.lines:
            for new_veh in self.new_departures(self._tcurrent, dt, lid):
                # Mark the Stop activity_type to done to start vehicle journey
                if new_veh.activity.activity_type is ActivityType.STOP:
                    new_veh.activity.is_done = True

                # If no user are waiting for this bus, switch activity_type to repositioning to end of line
                # if not new_veh.activities:
                #     stop_activity = new_veh.activity
                #     repo_activity = VehicleActivityRepositioning(stop_activity.node,
                #                                                  stop_activity.path,
                #                                                  stop_activity.user)
                #     new_veh.add_activities([repo_activity])
                #     new_veh.next_activity()

                log.info(f"Start {new_veh}")
                if self._observer is not None:
                    new_veh.attach(self._observer)

            self.clean_arrived_vehicles(lid)

    def periodic_maintenance(self, dt: Dt):
        pass

    def replanning(self):
        pass

    def rebalancing(self, next_demand: List[User], horizon: List[Vehicle]):
        pass

    def service_level_costs(self, nodes: List[str]) -> dict:
        """
                Returns the dict of costs representing the cost of the service computed from a path

                Args:
                    nodes: path (list of nodes)
        """
        return create_service_costs()

    def remove_activity_by_index(self, veh, index):
        """Method that removes an activity in a public transport vehicle plan by index and
        adapt the following activity path consequently.

        Args:
            -veh: public transport vehicle from which the activity should be removed from plan
            -index: index of the activity to remove in the list of vehicle's all activities
        """
        all_activities = [veh.activity] + list(veh.activities)
        assert len(all_activities) > index + 1, f'There should be an activity in '\
            'public transportation vehicle plan after a pickup/serving activity...'
        if index == 0:
            # Interrupt current activity and modify next activity consequently
            ind_curr_node_in_curr_act = veh.path_to_nodes(all_activities[0].path).index(veh._current_node)
            ongoing_leg = all_activities[0].path[ind_curr_node_in_curr_act:]
            ongoing_leg[0] = (ongoing_leg[0][0], veh._remaining_link_length)
            all_activities[1].modify_path(ongoing_leg + all_activities[1].path)
            veh.activity = None
        else:
            # Activity to remove has not begun, remove it from planning and update next activity consequently
            next_activity_new_path = all_activities[index].path + all_activities[index+1].path
            all_activities[index+1].modify_path(next_activity_new_path)
            del veh.activities[index-1]

    def remove_user_activities(self, user):
        """Method that removes the pick-up and serving activties related to a certain
        user in the plan of the vehicle this user is waiting for.

        Args:
            -user: user currently waiting a public transport vehicle but who finally
                   wont ride this vehicle
        """
        veh = user.waited_vehicle
        assert veh.mobility_service == self.id, f'User {user.id} is not waiting a public transport'\
            ' vehicle, wrong call of remove_user_activities method.'

        ## Remove user pickup activity
        all_activities = [veh.activity] + list(veh.activities)
        user_pu_act_ind = [i for i in range(len(all_activities)) if all_activities[i].user == user][0] # pickup is necessarily before serving
        self.remove_activity_by_index(veh, user_pu_act_ind)

        ## Remove user serving activity
        all_activities = [veh.activity] + list(veh.activities)
        user_serving_act_ind = [i for i in range(len(all_activities)) if all_activities[i].user == user][0]
        self.remove_activity_by_index(veh, user_serving_act_ind)

    def modify_user_drop_node(self, user, veh, new_drop_node, former_drop_node):
        """Method that modifies the drop node of a user who already appears in a
        PublicTransportMobilityService vehicle's plan (i.e. who has already been matched
        with the vehicle or already been picked up by it) by updating the vehicle's plan consequently.
        The order in which the line stops are visited is kept untouched while the order
        of activities of the oublic transport vehicle plan can be modified.

        Args:
            -user: user who wants to change her drop node
            -new_drop_node: the new drop node of the user
            -former_drop_node: the former drop node of the user
        """
        assert veh.mobility_service == self.id, f'Wrong call of modify_user_drop_node method: '\
            f'vehicle {veh.id} does not belong to {self.id} mobility service'
        if new_drop_node != former_drop_node:
            all_activities = [veh.activity] + list(veh.activities)
            user_serving_act_ind = [i for i in range(len(all_activities)) \
                if all_activities[i].user == user and type(all_activities[i]).__name__=='VehicleActivityServing']
            assert user_serving_act_ind, f'User {user.id} serving activity should appear'\
                f' in vehicle {veh} plan to be able to modify user drop node'
            user_serving_act_ind = user_serving_act_ind[0]
            # Step 1 = Remove the former serving activity for user
            self.remove_activity_by_index(veh, user_serving_act_ind)
            all_activities = [veh.activity] + list(veh.activities)

            # Step 2 = Insert the new serving activity for user
            # Find the new drop node in the PT vehicle's activities
            found = False
            for ind, a in enumerate(all_activities):
                if a is not None and type(a).__name__ != 'VehicleActivityStop':
                    try:
                        ind_new_drop_node = veh.path_to_nodes(a.path).index(new_drop_node)
                        found = True
                        break
                    except ValueError:
                        pass
            if not found:
                log.error(f'Could not find new drop node {new_drop_node} in vehicle {veh.id} plan')
                sys.exit(-1)
            # Deduce the activity to modify
            modified_act = all_activities[ind]
            if ind == 0:
                # Interrupt current activity, modify next activity consequently and insert user serving activity
                ind_curr_node_in_curr_act = veh.path_to_nodes(modified_act.path).index(veh._current_node)
                assert ind_curr_node_in_curr_act <= ind_new_drop_node, \
                    f'New drop node {new_drop_node} of user {user.id} has already been passed...'
                serving_act_path = modified_act.path[ind_curr_node_in_curr_act:ind_new_drop_node]
                modified_act_path = modified_act.path[ind_new_drop_node:]
                serving_act_path[0] = (serving_act_path[0][0], veh._remaining_link_length)
                serving_act = VehicleActivityServing(node=new_drop_node,
                                                    path=serving_act_path,
                                                    user=user)
                modified_act.modify_path(modified_act_path)
                veh.activity = None
                veh.activities.insert(0, serving_act)
                veh.activities.insert(1, modified_act)
            else:
                # Modify the activity and insert user serving activty before it in plan
                serving_act_path = modified_act.path[:ind_new_drop_node]
                modified_act_path = modified_act.path[ind_new_drop_node:]
                serving_act = VehicleActivityServing(node=new_drop_node,
                                                    path=serving_act_path,
                                                    user=user)
                modified_act.modify_path(modified_act_path)
                veh.activities.insert(ind-1, serving_act)
        else:
            # Former and new drop nodes are equal: there is nothing to do
            pass

    def modify_passenger_drop_node(self, passenger, new_drop_node, former_drop_node):
        """Method that modifies the drop node of a user which is already in a
        PublicTransportMobilityService vehicle by updating the vehicle's plan consequently.

        Args:
            -passenger: user in a public transport vehicle who wants to change her drop node
            -new_drop_node: the new drop node of the passenger
            -former_drop_node: the former drop node of the passenger
        """
        veh = passenger._vehicle
        self.modify_user_drop_node(passenger, veh, new_drop_node, former_drop_node)

    def __dump__(self):
        return {"TYPE": ".".join([PublicTransportMobilityService.__module__, PublicTransportMobilityService.__name__]),
                "ID": self.id}

    @classmethod
    def __load__(cls, data):
        new_obj = cls(data['ID'])
        return new_obj
