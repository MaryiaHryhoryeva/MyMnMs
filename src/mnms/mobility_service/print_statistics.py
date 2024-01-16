import numpy as np
from typing import Dict, List, Optional
from mnms.graph.layers import MultiLayerGraph
from mnms.demand.user import User, UserState
from mnms.time import Dt, Time
from mnms.mobility_service.ride_hailing import *
from mnms.mobility_service.ride_hailing_batch import *
from mnms.flow.MFD import MFDFlowMotor
from mnms.vehicles.veh_type import Vehicle, ActivityType
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os.path


#### if a user is refused, at the end of simulation, we cannot see to which company he belonged ############

#### if at the end of the simulation user's attribute 'refused' == 1 and
#### the attribute 'available_mobility_services' is empty - the user appeared but was not matched ####

#### if user's attribute 'refused' == 0, then the attribute 'available_mobility_services' is not empty ######
#### all users initially have 'refused' == 1, but with matching it changes to 'refused' == 0 ###############

#### if at the end of the simulation user's attribute 'refused' == 1 and
#### the attribute 'available_mobility_services' is not empty - the user appeared very close to the end of simulation
#### and didn't have time to be matched

def nb_of_cancellations():
    nb_cancel_n_wait = {}
    nb_real_cancel = {}
    for company in RideHailingServiceIdleCharge.instances:
        nb_cancel_n_wait[company.id] = company.refused_users_counter  # deadend users - not matched users
    print("Nb of refused users:")
    nb_refused = nb_cancel_n_wait.copy()
    print(nb_refused)
    for user in User.instances:
        if len(list(user.available_mobility_services)) > 0 and \
                user.refused == 1 and \
                list(user.available_mobility_services)[0] != 'PV':  # that are still waiting to be matched when the simulation is over
            nb_cancel_n_wait[list(user.available_mobility_services)[0]] += 1
    print("Nb of cancellations (refused + those who are still waiting to be matched when the simulation is over):")
    print(nb_cancel_n_wait)
    return nb_cancel_n_wait, nb_refused


def nb_of_users():  ## nb of cancelled users + nb of users with 0 in refused
    number_of_users = {}
    for company in RideHailingServiceIdleCharge.instances:
        number_of_users[company.id] = company.refused_users_counter
        for user in User.instances:
            if len(list(user.available_mobility_services)) > 0 and list(user.available_mobility_services)[0] == company.id:
                number_of_users[company.id] += 1
    print("Nb of users:")
    print(number_of_users)
    return number_of_users


def nb_of_served_users(number_of_users, number_of_cancellations):
    number_of_served_users = {}
    for key in number_of_users:
        number_of_served_users[key] = number_of_users[key] - number_of_cancellations[key]
    print("Nb of served users (nb of users-cancellations):")
    print(number_of_served_users)
    return number_of_served_users


def avg_matching_time():
    nb_of_matched_users = defaultdict(int)
    avg_match_time = {}
    for user in User.instances:
        if user.matched_time is not None:
            nb_of_matched_users[list(user.available_mobility_services)[0]] += 1

    for key2 in nb_of_matched_users:
        if nb_of_matched_users[key2] != 0:
            avg_match_time[key2] = 0
        for instance in User.instances:
            """print('\n')
            print(instance.id)
            print(instance.state)
            print(instance.matched_time)
            print(instance.refused)
            print(str(list(instance.available_mobility_services)))"""
            if instance.refused == 0 and list(instance.available_mobility_services)[0] == key2 and instance.matched_time is not None:
                avg_match_time[key2] += Dt.to_seconds(instance.matched_time) - Dt.to_seconds(instance.departure_time)

                # dummy = Dt.to_seconds(instance.matched_time) - Dt.to_seconds(instance.departure_time)
                # dummy = Dt.__sub__(instance.matched_time, instance.departure_time)

    for key3 in avg_match_time:
        avg_match_time[key3] = avg_match_time[key3] / nb_of_matched_users[key3]

    print("average_matching_time: " + str(avg_match_time))
    print("nb of matched users used to calculate the matching time: " + str(nb_of_matched_users))       # all users with non-empty matched time
    return avg_match_time


def avg_pickup_time():
    average_pickup_time = {}
    nb_of_picked_up_users = defaultdict(int)
    for user in User.instances:
        if user.picked_up_time is not None:
            nb_of_picked_up_users[list(user.available_mobility_services)[0]] += 1
    for key2 in nb_of_picked_up_users:
        if nb_of_picked_up_users[key2] != 0:
            average_pickup_time[key2] = 0
        for instance in User.instances:
            if instance.refused == 0 and list(instance.available_mobility_services)[0] == key2 and \
                    instance.state != UserState.WAITING_VEHICLE and instance.picked_up_time is not None:
                average_pickup_time[key2] += Dt.to_seconds(instance.picked_up_time) - Dt.to_seconds(
                    instance.matched_time)

    for key3 in average_pickup_time:
        average_pickup_time[key3] = average_pickup_time[key3] / nb_of_picked_up_users[key3]

    print("average_pickup_time: " + str(average_pickup_time))
    print("nb of picked up users used to calculate the pick up time: " + str(nb_of_picked_up_users))    # all users with non-empty pick-up time
    return average_pickup_time


def avg_idle_dist():
    average_idle_distance = {}
    total_idle_distance = {}
    total_service_distance = {}
    nb_of_picked_up_users = defaultdict(int)
    nb_of_served_users = defaultdict(int)
    for user in User.instances:
        if user.picked_up_time is not None:
            nb_of_picked_up_users[list(user.available_mobility_services)[0]] += 1

    for key2 in nb_of_picked_up_users:
        if nb_of_picked_up_users[key2] != 0:
            average_idle_distance[key2] = 0
        for instance in Vehicle.instances:
            if instance.mobility_service == key2 and nb_of_picked_up_users[key2] != 0:
                average_idle_distance[key2] += instance._pickup_distance

    for key3 in average_idle_distance:
        average_idle_distance[key3] = average_idle_distance[key3] / nb_of_picked_up_users[key3]
    print("average_idle_distance: " + str(average_idle_distance))

    for key2 in nb_of_picked_up_users:
        if nb_of_picked_up_users[key2] != 0:
            total_idle_distance[key2] = 0
        for instance in Vehicle.instances:
            if instance.mobility_service == key2 and nb_of_picked_up_users[key2] != 0:
                total_idle_distance[key2] += instance._pickup_distance
    print("total_idle_distance: " + str(total_idle_distance))

    for user in User.instances:
        if user.arrival_time is not None:
            nb_of_served_users[list(user.available_mobility_services)[0]] += 1

    for key2 in nb_of_served_users:
        if nb_of_served_users[key2] != 0:
            total_service_distance[key2] = 0
        for instance in Vehicle.instances:
            if instance.mobility_service == key2 and nb_of_served_users[key2] != 0:
                total_service_distance[key2] += instance._service_distance
    print("total_service_distance: " + str(total_service_distance))

    return average_idle_distance, total_idle_distance, total_service_distance

def profit_analysis():
    total_profit_per_comp = {}     # total profit of drivers of each company
    for company in RideHailingServiceIdleCharge.instances:
        total_profit_per_comp[company.id] = 0
    total_profit_per_comp['PV'] = 0
    for veh in Vehicle.instances:
        total_profit_per_comp[veh.mobility_service] += veh.driver_profit
    print("total profit of drivers of each company: " + str(total_profit_per_comp))

    nb_of_matched_users = defaultdict(int)
    for user in User.instances:
        if user.matched_time is not None:
            nb_of_matched_users[list(user.available_mobility_services)[0]] += 1

    avg_profit_per_trip = {}
    for key in total_profit_per_comp:
        avg_profit_per_trip[key] = total_profit_per_comp[key] / nb_of_matched_users[key]
    print("avg profit of driver per trip: " + str(avg_profit_per_trip))

    veh_count = {}
    avg_driver_profit_per_day = {}
    for company in RideHailingServiceIdleCharge.instances:
        veh_count[company.id] = 0
    veh_count['PV'] = 0
    for veh in Vehicle.instances:
        veh_count[veh.mobility_service] += 1
    for key in veh_count:
        avg_driver_profit_per_day[key] = total_profit_per_comp[key] / veh_count[key]
    print("avg profit of driver per day: " + str(avg_driver_profit_per_day))
    return avg_profit_per_trip, avg_driver_profit_per_day, total_profit_per_comp, veh_count


def demand_analysis():
    df = pd.read_csv('/Users/maryia/Documents/GitHub/MnMS/examples/my_example/demand_generated.csv', sep=';')
    uber_demands_regions = df.loc[df['SERVICE'] == 'UBER', 'D_dem_lvl']
    lyft_demands_regions = df.loc[df['SERVICE'] == 'LYFT', 'D_dem_lvl']
    print("uber demand destination distribution over regions:\n" + str(uber_demands_regions.value_counts(normalize=True)))
    print("lyft demand destination distribution over regions:\n" + str(lyft_demands_regions.value_counts(normalize=True)))


def avg_speed():
    speed_per_res_mean = {}
    speed_per_res = {}
    df = pd.read_csv('/Users/maryia/Documents/GitHub/MnMS/examples/my_example/MFD_info.csv', sep=';')
    # print(list(df.columns.values))
    # print(df["RESERVOIR"].unique())
    # print(df["SPEED"].unique())
    # print(df["SPEED"].sum())
    # print(len(df["SPEED"]))
    for i in df["RESERVOIR"].unique():
        # print(df.loc[df['RESERVOIR'] == i, 'SPEED'].sum())
        speed_per_res_mean[i] = df.loc[df['RESERVOIR'] == i, 'SPEED'].mean()
        speed_per_res[i] = df.loc[df['RESERVOIR'] == i, 'SPEED']
    print("Avg speed per reservoir: ")
    print(speed_per_res_mean)

    FIG_SIZE_HALF = (6, 3)
    FONT_SIZE_LAB = 14
    FONT_SIZE_LEG = 12
    FONT_SIZE_AXI = 12

    labels = ["17", "18", "19", "20"]
    plt.subplots(1, 1, constrained_layout=True)

    plt.subplot(1, 1, 1)
    #data1 = [val for val in speed_per_res['RES'] for _ in (0, 1)]

    plt.plot(speed_per_res['RES'], 'magenta')

    plt.ylabel("Network speed (m/s)", fontsize=FONT_SIZE_LAB)
    plt.xlabel("Time (h)", fontsize=FONT_SIZE_LAB)
    plt.xticks(np.linspace(0, len(speed_per_res['RES']), len(labels)), labels, fontsize=FONT_SIZE_AXI)
    plt.yticks(fontsize=FONT_SIZE_AXI)
    #plt.title('(a) Arriving demand requests (same for both companies)')
    #plt.legend(loc="lower right")
   # plt.show()


    return speed_per_res_mean

def save_idle_dist():
    idle_distance = {}
    nb_of_picked_up_users = defaultdict(int)
    for user in User.instances:
        if user.picked_up_time is not None:
            nb_of_picked_up_users[list(user.available_mobility_services)[0]] += 1

    for key2 in nb_of_picked_up_users:
        if nb_of_picked_up_users[key2] != 0:
            idle_distance[key2] = []
        for instance in Vehicle.instances:
            if instance.mobility_service == key2 and nb_of_picked_up_users[key2] != 0:
                idle_distance[key2].append(instance._pickup_distance)
    data = {
        "UBER": idle_distance['UBER'],
        "LYFT": idle_distance['LYFT']
    }

    folder = '/Users/maryia/Documents/GitHub/MnMS/examples/my_example/graphs/Decentr_random_vs_donut/'
    file_name = 'idle_dist_random_0_150.csv'
    df = pd.DataFrame(data)
    # print(df)
    if not os.path.exists(folder + file_name):
        df.to_csv(folder + file_name, sep=';', index=False)
    else:
        df.to_csv(folder + 'dummy_file', sep=';', index=False)



def statistics_output(nb_users, nb_veh, nb_refused, nb_cancel_n_wait, avg_match_time, avg_pick_time,
                      avg_idle, total_idle, total_service,
                      profit_per_trip, profit_per_day, total_profit_per_comp,
                      speed):
    service = ['UBER', 'LYFT']
    nb_users2 = [nb_users['UBER'], nb_users['LYFT']]
    nb_veh2 = [nb_veh['UBER'], nb_veh['LYFT']]

    charge = []
    for company in RideHailingServiceIdleCharge.instances:
        if company.id == 'UBER':
            charge.append(company.idle_km_or_h_charge)
            break
    for company in RideHailingServiceIdleCharge.instances:
        if company.id == 'LYFT':
            charge.append(company.idle_km_or_h_charge)
            break

    print(charge)
    nb_of_cancel2 = [nb_refused['UBER'], nb_refused['LYFT']]
    nb_cancel_n_wait2 = [nb_cancel_n_wait['UBER'], nb_cancel_n_wait['LYFT']]
    avg_match_time2 = [avg_match_time['UBER'], avg_match_time['LYFT']]
    avg_pick_time2 = [avg_pick_time['UBER'], avg_pick_time['LYFT']]
    avg_wait_time = [g + h for g, h in zip(avg_match_time2, avg_pick_time2)]
    avg_idle2 = [avg_idle['UBER'], avg_idle['LYFT']]
    total_idle2 = [total_idle['UBER'], total_idle['LYFT']]
    total_service2 = [total_service['UBER'], total_service['LYFT']]

    total_dist = []
    for i in range(len(total_service2)):
        total_dist.append(total_idle2[i] + total_service2[i])
    idle_vs_total = []
    for i in range(len(total_dist)):
        idle_vs_total.append((total_idle2[i] / total_dist[i]) * 100)

    profit_per_trip2 = [profit_per_trip['UBER'], profit_per_trip['LYFT']]
    profit_per_day2 = [profit_per_day['UBER'], profit_per_day['LYFT']]
    total_profit_per_comp2 = [total_profit_per_comp['UBER'], total_profit_per_comp['LYFT']]

    total_charge = []
    for i in range(2):
        total_charge.append((total_idle2[i] / 1000) * charge[i])
    speed2 = [list(speed.values())[0], list(speed.values())[0]]

    data = {
        "SERVICE": service,
        "NB_USERS": nb_users2,
        "NB_VEH": nb_veh2,
        "CHARGE": charge,
        "CANCEL": nb_of_cancel2,
        "CANCEL_N_WAIT": nb_cancel_n_wait2,
        "MATCH_TIME": avg_match_time2,
        "PICKUP_TIME": avg_pick_time2,
        "WAIT_TIME": avg_wait_time,
        "AVG_IDLE": avg_idle2,
        "IDLE_VS_TOTAL": idle_vs_total,
        "TOTAL_IDLE": total_idle2,
        "TOTAL_SERVICE": total_service2,
        "PROFIT_PER_TRIP": profit_per_trip2,
        "PROFIT_PER_DAY": profit_per_day2,
        "TOTAL_PROFIT": total_profit_per_comp2,
        "TOTAL_CHARGE": total_charge,
        "SPEED": speed2
    }

    """folder = '/Users/maryia/Documents/GitHub/MnMS/examples/my_example/graphs/Random_seed/data/'
    file_name = '40_150_150.csv'
    df = pd.DataFrame(data)
    #print(df)
    if not os.path.exists(folder + file_name):
        df.to_csv(folder + file_name, sep=';', index=False)
    else:
        df.to_csv(folder + 'dummy_file', sep=';', index=False)"""

