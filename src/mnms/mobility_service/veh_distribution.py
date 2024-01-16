import random
import pandas as pd
import os

def generate_rh_supply_scenario(rh_service, nb_veh):
    avs = []
    for i in range(nb_veh):
        # Draw a random node where AV starts
        node = random.choice(list(rh_service.layer.graph.nodes.keys()))
        avs.append([node])
    rh_supply = pd.DataFrame(avs, columns=['NODE'])
    if not os.path.isfile('rh_veh_pos.csv'):
        rh_supply.to_csv('rh_veh_pos.csv', sep=';', index=False)
    else:
        print(f"A file w/ vehicles' positions already exist. Nothing is generated to prevent overwriting.")

def create_rh_veh_pos(rh_service, file):
    with open(file, 'r'):
        df = pd.read_csv(file, delimiter=';', quotechar='|')
        [rh_service.create_waiting_vehicle(n) for n in df.NODE]




