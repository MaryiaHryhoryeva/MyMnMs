from typing import Tuple, Dict

import numpy as np
import gurobipy as grb
import pandas as pd
from hipop.shortest_path import dijkstra

from mnms import create_logger
from mnms.demand import User
from mnms.mobility_service.abstract import AbstractMobilityService
from mnms.time import Dt
from mnms.tools.exceptions import PathNotFound
from mnms.vehicles.veh_type import ActivityType, VehicleActivityServing, VehicleActivityStop, \
    VehicleActivityPickup, VehicleActivityRepositioning, Vehicle
from mnms.time import Time

log = create_logger(__name__)

class RideHailing_Batch(AbstractMobilityService):

    instances = []

    def __init__(self,
                 _id: str,
                 dt_matching: int = 60,  ## in seconds, indicates how often we perform matching (eg, to indicate that we match users&vehicles every 1 min)
                 dt_step_maintenance: int = 0) -> object:
        super(RideHailing_Batch, self).__init__(_id, 1, dt_matching, dt_step_maintenance)

        self.__class__.instances.append(self)
        self.gnodes = dict()
        self.nb_of_users_counter = 4
        self.refused_users_counter = 0
        self.max_pickup_dist = 2000                # maximum tolerable pickup distance for a driver
        self.max_pickup_time = Time("00:10:00")    # max tolerable pickup time for a driver
        self.cancellation_mode = 0                 #

        ####### Profits and costs ###########

        self.min_trip_price = 7
        self.service_km_profit = 1.7
        self.expenses_per_km = 0.3     # gaz + insurance + depreciation price
        self.idle_km_or_h_charge = 2
        self.driver_hour_min_payment = 18
        self.company_fee = 0.25          # percentage of profit that company takes from driver
        self.bonus = 0
        self.penalty = 0
        self.min_hour_earning_per_veh = 18

    def create_waiting_vehicle(self, node: str):
        assert node in self.graph.nodes
        new_veh = self.fleet.create_vehicle(node,
                                            capacity=self._veh_capacity,
                                            activities=[VehicleActivityStop(node=node)])
        new_veh.set_position(self.graph.nodes[node].position)

        if self._observer is not None:
            new_veh.attach(self._observer)

    def step_maintenance(self, dt: Dt):
        self.gnodes = self.graph.nodes

    def request(self, user: User, drop_node: str) -> tuple[Dt, float]:
        """
                Args:
                    user: User requesting a ride
                    drop_node:
                Returns: waiting time before pick-up
                """

        upos = user.position
        uid = user.id
        utility_per_user = []
        vehs = list(self.fleet.vehicles.keys())
        idle_service_dt = Dt(hours=24)
        occupied_service_dt = Dt(hours=24)
        while vehs:
            # Search for the nearest vehicle to the user
            #veh_pos = np.array([self.fleet.vehicles[v].position for v in vehs])
            #dist_vector = np.linalg.norm(veh_pos - upos, axis=1)
            #nearest_veh_index = np.argmin(dist_vector)
            #nearest_veh = vehs[nearest_veh_index]
            #vehs.remove(nearest_veh)
            for v in range(len(vehs)):
                choosen_veh = self.fleet.vehicles[vehs[v]]
                #            if not choosen_veh.is_full:
                if choosen_veh.is_empty:
                    # Vehicle available if either stopped or repositioning, and has no activity planned afterwards
                    available = True if ((choosen_veh.activity_type in [ActivityType.STOP, ActivityType.REPOSITIONING]) and (
                        not choosen_veh.activities)) else False
                    if available:
                        # Compute pick-up path and cost from end of current activity
                        veh_last_node = choosen_veh.activity.node if not choosen_veh.activities else \
                            choosen_veh.activities[-1].node
                        veh_path_idle, cost_idle = dijkstra(self.graph, veh_last_node, user.current_node, 'travel_time',
                                                  {self.layer.id: self.id}, {self.layer.id})
                        # If vehicle cannot reach user, skip and consider next vehicle
                        if cost_idle == float('inf'):
                            continue
                        len_path_idle = 0           # idle distance in meters
                        for i in range(len(veh_path_idle) - 1):
                            j = i + 1
                            len_path_idle += self.gnodes[veh_path_idle[i]].adj[veh_path_idle[j]].length

                        #print("\n")
                        #print(uid)
                        #print("Idle distance info:")
                        #print(cost_idle)
                        #print(len_path_idle)

                        ##############################
                        veh_path_service, cost_service = dijkstra(self.graph, user.current_node, user.path.nodes[-2], 'travel_time',
                                                            {self.layer.id: self.id}, {self.layer.id})
                        len_path_service = 0                # service distance in meters
                        for i in range(len(veh_path_service) - 1):
                            j = i + 1
                            len_path_service += self.gnodes[veh_path_service[i]].adj[veh_path_service[j]].length

                        #print("Service distance info:")
                        #print(cost_service)
                        #print(len_path_service)
                        ###############################
                        ##### per km charge:
                        min_price = self.min_trip_price
                        driver_profit = (len_path_service * self.service_km_profit) / 1000




                        expenses_per_km = ((len_path_idle + len_path_service) * self.expenses_per_km) / 1000
                        idle_charge = (len_path_idle * self.idle_km_or_h_charge) / 1000

                        total_profit = max(driver_profit, min_price) - expenses_per_km - idle_charge

                        ##### per hour charge:
                        #total_profit = self.matching_profit + (len_path_service * self.service_km_profit) / 1000 - \
                        #               ((len_path_idle + len_path_service) * self.expenses_per_km) / 1000 - \
                        #               (cost_idle * self.idle_km_or_h_charge) / 3600
                        #threshold = ((cost_idle + cost_service) * self.driver_hour_min_payment) / 3600

                        if user.dest_region_demand_level[0] == '1':
                            total_profit = total_profit + self.bonus

                        if user.dest_region_demand_level[0] == '-1':
                            total_profit = total_profit - self.penalty

                        else:
                            total_profit = total_profit

                        # idle_service_dt = Dt(seconds=len_path / choosen_veh.speed)
                        idle_service_dt = Dt(seconds=cost_idle)  # idle time (time needed to pickup a user)
                        #if user.pickup_dt[self.id] > idle_service_dt:

                        continue
        return idle_service_dt, total_profit


    def batch_matching(self):

        vehs = list(self.fleet.vehicles.keys())
        users = list(self._user_buffer.items())
        u_matrix = np.zeros((len(vehs), len(users)))
        matching = {}
        for v, veh in enumerate(vehs):
            for u, user in enumerate(users):
                if self.fleet.vehicles[veh].is_empty:
                    # Vehicle available if either stopped or repositioning, and has no activity planned afterwards
                    available = True if ((self.fleet.vehicles[veh].activity_type in [ActivityType.STOP, ActivityType.REPOSITIONING]) and (
                        not self.fleet.vehicles[veh].activities)) else False
                    if available:
                        veh_last_node = self.fleet.vehicles[veh].activity.node if not self.fleet.vehicles[veh].activities else \
                            self.fleet.vehicles[veh].activities[-1].node
                        veh_path_idle, cost_idle = dijkstra(self.graph, veh_last_node, user[1][0].current_node, 'travel_time',
                                                            {self.layer.id: self.id}, {self.layer.id})

                        # If vehicle cannot reach user, skip and consider next vehicle
                        if cost_idle == float('inf'):
                            u_matrix[v][u] = -1000
                            continue

                        idle_service_dt = Dt(seconds=cost_idle)  # idle time (time needed to pickup a user)
                        # If the match violates the pickup waiting time of user, skip and consider next vehicle
                        if user[1][0].pickup_dt[self.id] < idle_service_dt:   # if user time tolerance is below the actual pickup time
                            u_matrix[v][u] = -1000
                            continue

                        len_path_idle = 0  # idle distance in meters
                        for i in range(len(veh_path_idle) - 1):
                            j = i + 1
                            len_path_idle += self.gnodes[veh_path_idle[i]].adj[veh_path_idle[j]].length

                        veh_path_service, cost_service = dijkstra(self.graph, user[1][0].current_node, user[1][0].path.nodes[-2],
                                                                  'travel_time',
                                                                  {self.layer.id: self.id}, {self.layer.id})
                        len_path_service = 0  # service distance in meters
                        for i in range(len(veh_path_service) - 1):
                            j = i + 1
                            len_path_service += self.gnodes[veh_path_service[i]].adj[veh_path_service[j]].length

                        min_price = self.min_trip_price
                        driver_profit = (len_path_service * self.service_km_profit) / 1000

                        expenses_per_km = ((len_path_idle + len_path_service) * self.expenses_per_km) / 1000
                        idle_charge = (len_path_idle * self.idle_km_or_h_charge) / 1000

                        total_profit = max(driver_profit, min_price) - expenses_per_km - idle_charge
                        if user[1][0].dest_region_demand_level[0] == '1':
                            total_profit = total_profit + self.bonus

                        if user[1][0].dest_region_demand_level[0] == '-1':
                            total_profit = total_profit - self.penalty

                        else:
                            total_profit = total_profit

                        u_matrix[v][u] = total_profit


        if len(u_matrix) > 0 and len(u_matrix[0]) > 0:
            matching = self.ilp_gurobi(u_matrix)        # matching is dictionary where keys are indexes of vehicles and values are indexes of users

        # calculating the total profit from the matching
        matching_profit = 0
        for key, value in matching.items():
            matching_profit = matching_profit + u_matrix[key][value]

        # replacing zero and negative values in utility matrix with nan
        if len(matching) != 0:
            for c in range(len(u_matrix)):
                for r in range(len(u_matrix[0])):
                    if u_matrix[c][r] <= 0:
                        u_matrix[c][r] = np.nan

        # in the next part of code we calculate the time duration of trips for each matched pair (pickup+service time in seconds)
        trip_time_of_each_pair = {}
        for key, value in matching.items():
            veh_last_node = self.fleet.vehicles[vehs[key]].activity.node if not self.fleet.vehicles[vehs[key]].activities else \
                self.fleet.vehicles[vehs[key]].activities[-1].node
            veh_path_idle, cost_idle = dijkstra(self.graph, veh_last_node, users[value][1][0].current_node, 'travel_time',
                                                                {self.layer.id: self.id}, {self.layer.id})
            veh_path_service, cost_service = dijkstra(self.graph, user[1][0].current_node, user[1][0].path.nodes[-2],
                                                      'travel_time',
                                                      {self.layer.id: self.id}, {self.layer.id})
            trip_time_of_each_pair[(key, value)] = cost_idle + cost_service  # in seconds

        # matching contains the indexes of veh/users from u_matrix that we want to pair
        # if total profit is less than expected and matching is not empty -> remove the least profitable matching until the total profit becomes acceptable or no matching pairs left
        # expected profit = sum of time of all matched trips multiplied by min acceptable hour profit
        if len(matching) > 0:
            while matching_profit <= (sum(trip_time_of_each_pair.values())/3600) * self.min_hour_earning_per_veh:
                if len(matching) <= 0:
                    break
                profit_from_each_pair = {}
                for key, value in matching.items():
                    profit_from_each_pair[(key, value)] = u_matrix[key][value]      # get profit from each pair of veh/user
                #print("/////////")
                #print(matching)
                #print(profit_from_each_pair)
                #print(trip_time_of_each_pair)
                #print((sum(trip_time_of_each_pair.values())/3600) * self.min_hour_earning_per_veh)
                #print("/////////")
                idx_of_min_profit_value = min(profit_from_each_pair, key=profit_from_each_pair.get)     # get the veh/user indexes of the min profit
                matching_profit = matching_profit - u_matrix[idx_of_min_profit_value[0]][idx_of_min_profit_value[1]]    # reduce the profit by removed pair profit
                u_matrix[idx_of_min_profit_value[0]][idx_of_min_profit_value[1]] = np.nan       # delete the profit value from utility matrix
                del matching[idx_of_min_profit_value[0]]        # delete pair from matching
                del trip_time_of_each_pair[(idx_of_min_profit_value[0], idx_of_min_profit_value[1])]    # delete their trip time


        # perform final matching
        for key, value in matching.items():
            veh_last_node = self.fleet.vehicles[vehs[key]].activity.node if not self.fleet.vehicles[vehs[key]].activities else \
                self.fleet.vehicles[vehs[key]].activities[-1].node
            veh_path_idle, cost_idle = dijkstra(self.graph, veh_last_node, users[value][1][0].current_node, 'travel_time',
                                                                {self.layer.id: self.id}, {self.layer.id})



            self._cache_request_vehicles[users[value][0]] = self.fleet.vehicles[vehs[key]], veh_path_idle
            self.matching(users[value][1][0], users[value][1][1])
            self.profit_update(users[value][1][0], u_matrix[key][value])
            self._user_buffer.pop(users[value][0])
        self._cache_request_vehicles = dict()


    def ilp_gurobi(self, C, with_mip_start=False):
        assert len(C) > 0 and len(C[0]) > 0, 'Matrix passed to ilp method should ' \
                                             'not be empty or a vector.'
        if with_mip_start:
            # Compute a feasible integer solution with a heursitic to provide this
            # as a starting point to the branch and cut
            greedy_matching = self.greedy_assignment(C)
        n = len(C)  # nb of AVs
        m = len(C[0])  # nb of reqs
        set_I = range(n)  # AVs
        set_J = range(m)  # reqs
        c = {(i, j): C[i][j] for i in set_I for j in set_J}
        ## Create model
        opt_model = grb.Model(name="Matching Model")
        opt_model.Params.LogToConsole = 0
        ## Set decision variables
        # x_vars  = {(i,j): opt_model.addVar(vtype=grb.GRB.BINARY,
        #                name=f"x_{i}_{j}") for i in set_I for j in set_J}
        x_vars = opt_model.addVars(n, m, vtype=grb.GRB.BINARY)
        ## Set constraints
        constraints_j = {j: opt_model.addConstr(lhs=grb.quicksum(x_vars[i, j] for i in set_I),
                                                sense=grb.GRB.LESS_EQUAL,
                                                rhs=1,
                                                name=f"constraint_j_{j}") for j in set_J}
        constraints_i = {i: opt_model.addConstr(lhs=grb.quicksum(x_vars[i, j] for j in set_J),
                                                sense=grb.GRB.LESS_EQUAL,
                                                rhs=1,
                                                name=f"constraint_i_{i}") for i in set_I}
        # contraints_equality = {(i,j): opt_model.addConstr(lhs=x_vars[i,j],
        #    sense=grb.GRB.EQUAL,
        #    rhs=0,
        #    name=f"constraint_i_{i}") for i in set_I for j in set_J if c[i,j] < 0}
        ## Set objective
        objective = grb.quicksum(x_vars[i, j] * c[i, j] for i in set_I for j in set_J)
        opt_model.ModelSense = grb.GRB.MAXIMIZE
        opt_model.setObjective(objective)
        ## Enlight with initial solution
        if with_mip_start:
            start_values = {f'x_{i}_{j}': 0 if (i not in greedy_matching.keys()) or (greedy_matching[i] != j) else 1 for
                            i in set_I for j in set_J}
            for v in opt_model.getVars():
                v.Start = start_values[v.name]

        ## Call the solver
        opt_model.optimize()

        ## Get results
        opt_df = pd.DataFrame.from_dict(x_vars, orient='index', columns=['variable_object'])
        opt_df.reset_index(inplace=True)
        opt_df['solution_value'] = opt_df['variable_object'].apply(lambda item: item.X)
        opt_df.drop(columns=['variable_object'], inplace=True)

        ## Deduce matching dict
        # Filter ones
        opt_df = opt_df[opt_df['solution_value'] == 1]
        matching = opt_df['index'].tolist()
        matching = dict(matching)

        return matching


    def matching(self, user: User, drop_node: str):
        veh, veh_path = self._cache_request_vehicles[user.id]
        upath = list(user.path.nodes)
        upath = upath[upath.index(user._current_node):upath.index(drop_node) + 1]
        user_path = self.construct_veh_path(upath)
        veh_path = self.construct_veh_path(veh_path)
        activities = [
            VehicleActivityPickup(node=user._current_node,
                                  path=veh_path,
                                  user=user),
            VehicleActivityServing(node=drop_node,
                                   path=user_path,
                                   user=user)
        ]

        veh.add_activities(activities)
        user.set_state_waiting_vehicle()

        if veh.activity_type is ActivityType.STOP:
            veh.activity.is_done = True

    def profit_update(self, user: User, driver_profit_per_trip: float):
        veh, veh_path = self._cache_request_vehicles[user.id]
        veh.trip_counter_update()
        veh.driver_profit_update(driver_profit_per_trip)

    def launch_matching(self):
        """
                Method that launch passenger-vehicles matching, through 1. requesting and 2. matching.
                Returns: empty list # TODO - should be cleaned

                """
        # refuse_user = list()

        if self._counter_matching == self._dt_matching:     # counter_matching checks when we performed the matching last time as it should be every "dt_matching" time
            self._counter_matching = 0
            self.batch_matching()

            """for uid, (user, drop_node) in list(self._user_buffer.items()):
                # User makes service request
                service_dt, driver_profit_per_trip = self.request(user, drop_node)

                if user.pickup_dt[self.id] > service_dt:
                    # If pick-up time is below passengers' waiting tolerance
                    # Match user with vehicle
                    self.matching(user, drop_node)
                    self.profit_update(user, driver_profit_per_trip)

                    # Remove user from list of users waiting to be matched
                    self._user_buffer.pop(uid)
                else:
                    # If pick-up time exceeds passengers' waiting tolerance
                    log.info(f"{uid} refused {self.id} offer (predicted pickup time too long)")
                    #user.set_state_stop()
                    #user.notify(self._tcurrent)
                    #refuse_user.append(user)
                self._cache_request_vehicles = dict()"""

            #self._user_buffer = dict()
            # NB: we clean _user_buffer here because answer provided by the mobility
            #     service should be YES I match with you or No I refuse you, but not
            #     let's wait the next timestep to see if I can find a vehicle for you
            #     Mob service has only one chance to propose a match to the user,
            #     except if user request the service again
        else:
            self._counter_matching += 1

        return list()  # refuse_user        # list of refused users?

    def __dump__(self):
        return {"TYPE": ".".join([RideHailing_Batch.__module__, RideHailing_Batch.__name__]),
                "DT_MATCHING": self._dt_matching,
                "VEH_CAPACITY": self._veh_capacity,
                "ID": self.id}

    @classmethod
    def __load__(cls, data):
        new_obj = cls(data['ID'], data["DT_MATCHING"], data["VEH_CAPACITY"])
        return new_obj

