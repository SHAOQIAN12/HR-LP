import ijson
import json
import math
from random import sample,seed,shuffle
import pandas as pd
import numpy as np
#from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from os import path
from math import ceil
from scipy.spatial.distance import cdist, euclidean
#warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def zone_supplement (dataRoute,route,stop):

    if route == 'RouteID_539fc41a-4ef8-43bb-9bc6-cddc7ed4379b' and stop == 'WW':  # fix the wrong zone id
        tempNode = closestNode('RouteID_539fc41a-4ef8-43bb-9bc6-cddc7ed4379b', 'WW', dataRoute)
        stop = tempNode
        zone1 = dataRoute['RouteID_539fc41a-4ef8-43bb-9bc6-cddc7ed4379b'].get('stops').get(stop).get('zone_id')
    elif route == 'RouteID_02f91cfa-3839-4e55-91a0-ab19c3e77683' and stop == 'XQ':  # fix the wrong zone id
        tempNode = closestNode('RouteID_02f91cfa-3839-4e55-91a0-ab19c3e77683', 'XQ', dataRoute)
        stop = tempNode
        zone1 = dataRoute['RouteID_02f91cfa-3839-4e55-91a0-ab19c3e77683'].get('stops').get(stop).get('zone_id')
    else:
        zone1 = dataRoute[route]['stops'][stop]['zone_id']
    if pd.isnull(zone1):
        if dataRoute[route]['stops'][stop]['type'] == 'Station':
            zone1 = dataRoute[route]['station_code']
        else:
            tempNode = closestNode(route, stop,dataRoute)
            stop = tempNode
            zone1 = dataRoute[route].get('stops').get(stop).get('zone_id')
    return zone1


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def zone_set(dataRoute,route):
    zone_dict={}
    result= []
    for i in dataRoute[route]['stops'].keys():
        zone1 = zone_supplement(dataRoute,route,i)
        zone_dict[i] = zone1
        if zone1 not in result:
            result.append(zone1)
    zone_dict_1 = {}
    for i in result:
        zone_dict_1[i]= get_key(zone_dict,i)
    return zone_dict_1


def closestNode(route, node,route_data):
    x_node = route_data[route].get('stops').get(node).get('lat')
    y_node = route_data[route].get('stops').get(node).get('lng')
    min_node = 99999999
    which_node = ''  # the error is handled in calling arguments
    for n in route_data[route].get('stops'):
        if n == node or not isinstance(route_data[route].get('stops').get(n).get('zone_id'), str):
            continue
        x1 = route_data[route].get('stops').get(n).get('lat')
        y1 = route_data[route].get('stops').get(n).get('lng')
        dist_node = math.sqrt((x1 - x_node) * (x1 - x_node) + (y1 - y_node) * (y1 - y_node))
        if dist_node < min_node and dist_node > math.exp(-15):
            min_node = dist_node
            which_node = n
    return which_node

def avg_travel_time(zone1,zone2,tra_time_1):
    tra_time = 0
    n = 0
    for i in zone1:
        for j in zone2:
            tra_time += tra_time_1[i][j]
            n +=1
    return round(tra_time/n,4)


def cal_centroid(zone,route_data,route):
    x_stop = []
    y_stop = []
    for j in zone:
        x_stop.append(route_data[route]['stops'][j]['lat'])
        y_stop.append(route_data[route]['stops'][j]['lng'])
    x_centroid = (sum(x_stop) / len(x_stop))
    y_centroid = (sum(y_stop) / len(y_stop))
    a = [x_centroid, y_centroid]
    return a


# calculate the average distance between centroid of zones
def avg_tra_distance(zone1,zone2,route_data,route):
    n = 0
    for i in zone1:
        vector1 = np.array([route_data[route]['stops'][i]['lat'], route_data[route]['stops'][i]['lng']])
        for j in zone2:
            vector2 = np.array([route_data[route]['stops'][j]['lat'], route_data[route]['stops'][j]['lng']])
            n +=1
    cen_vector1 = np.array(cal_centroid(zone1,route_data,route))
    cen_vector2 = np.array(cal_centroid(zone2,route_data,route))
    centroid_dist = np.linalg.norm(cen_vector1 - cen_vector2, ord=1)
    return centroid_dist


def get_station(route_data,route):
    for i in route_data[route]['stops'].keys():
        if route_data[route]['stops'][i]['type'] == 'Station':
            station = i
    return station



# calculate the average travel time from depot to stops in zone
def depot_zone(zone,station,tra_time_1):
    tra_time = 0
    n = 0
    for i in zone:
        tra_time += tra_time_1[station][i]
        n +=1
    return round(tra_time/n,4)

# calculate the average travel time from stops in zone to depot
def zone_depot(zone,station,tra_time_1):
    tra_time = 0
    n = 0
    for i in zone:
        tra_time += tra_time_1[i][station]
        n +=1
    return round(tra_time/n,4)


def split(zone_id):
    split1 = zone_id.split("-")
    P1 = split1[0]
    split2 = split1[1].split(".")
    P2 = split2[0]
    res = []
    res[:] = split2[1]
    P3 = res[0]
    P4 = res[1]
    result = [P1,P2,P3,P4]
    return result


def check_main_zone(zone1,zone2,station):
    if zone1 == station or zone2 == station:
        result = 1
    else:
        zone1_split = split(zone1)
        zone2_split = split(zone2)
        if zone1_split[0]==zone2_split[0] and zone1_split[1]==zone2_split[1]:
            result = 0
        else: result = 1
    return result



def data_load_apply(path):
    with open(path + "new_route_data.json", 'rb') as file:
        route_data = json.load(file)
    routes = list(route_data.keys())
    return routes,route_data



def process_zone_info(BASE_DIR, station):
    path1 = BASE_DIR + '/data/model_build_outputs/'
    with open(BASE_DIR + '/data/model_build_inputs_' + station + "/route_data.json", 'rb') as file:
        route_data = json.load(file)
    with open(BASE_DIR + '/data/model_build_inputs_' + station + '/travel_times.json', 'rb') as file:
        travel_time_data = json.load(file)

    route = list(route_data.keys())
    zone_ = {}
    for i in range(len(route)):
        #with open(path1 + route[i] +".json", 'rb') as file:
            #tra_time = json.load(file)
        tra_time = travel_time_data[route[i]]
        with open(path1 + route[i] + '.json', 'w') as file:
            json.dump(tra_time, file)
        station_code = route_data[route[i]]['station_code']  # station code, e.g. DLA4
        zone_[i] = {}
        zone_[i]['station'] = station_code
        zone_[i]['stop'] = []
        zone_info = {}
        zone_info['station'] = route_data[route[i]]['station_code']
        zone_info['avg_tra_time'] = {}
        zone_info['avg_centroid'] = {}
        zone_info['depot_zone_tra_time_ratio'] = {}
        zone_info['zone_depot_tra_time_ratio'] = {}
        zone_info['P1P2'] = {}
        zone_dict = zone_set(route_data,route[i])

        zone_list = list(zone_dict.keys())
        for zone in zone_list:
            zone_[i]['stop'].append(len(zone_dict[zone]))
        station = get_station(route_data,route[i])
        for zone1 in zone_list:
            zone_info['avg_tra_time'][zone1] = {}
            zone_info['avg_centroid'][zone1] = {}
            zone_info['depot_zone_tra_time_ratio'][zone1] = {}
            zone_info['zone_depot_tra_time_ratio'][zone1] = {}
            zone_info['P1P2'][zone1] = {}
            for zone2 in zone_list:
                if zone1 == zone2:
                    zone_info['avg_tra_time'][zone1][zone2] = 0
                    zone_info['avg_centroid'][zone1][zone2] = 0
                    zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = 0
                    zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = 0
                    zone_info['P1P2'][zone1][zone2] = 0
                else:
                    zone_info['avg_tra_time'][zone1][zone2] = avg_travel_time(zone_dict[zone1],zone_dict[zone2],tra_time)
                    zone_info['avg_centroid'][zone1][zone2] = avg_tra_distance(zone_dict[zone1], zone_dict[zone2], route_data, route[i])
                    if zone1 == station_code or zone2 == station_code:
                        zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = 0
                        zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = 0
                    else:
                        zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = depot_zone(zone_dict[zone2],station,tra_time)/depot_zone(zone_dict[zone1],station,tra_time)
                        zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = zone_depot(zone_dict[zone2], station, tra_time) /zone_depot(zone_dict[zone1], station, tra_time)
                    result = check_main_zone(zone1,zone2,station_code)
                    zone_info['P1P2'][zone1][zone2] = result

        with open(path1 + route[i] + 'zone.json', 'w') as file:
            json.dump(zone_info, file)
    with open(path1 + str(station_code) + '.json', 'w') as file:
        json.dump(zone_, file)
    
def process_tra_time_apply(path, path2):
    with open(path + "new_route_data.json", 'rb') as fil:
        route_data = json.load(fil)

    routes = list(route_data.keys())
    n_routes = len(routes)

    fil = open(path + "new_travel_times.json", 'rb')
    parser = ijson.parse(fil, use_float=True)

    for i in range(n_routes):
        flag = 0
        comma_flag = False
        while True:
            if flag == 0:
                while True:
                    prefix, event, value = next(parser)
                    # print("Prefixo =", prefix)
                    # print("Evento =",event)
                    # print("Value = ", value)
                    if prefix != "":
                        flag = 1
                        route_ID = prefix
                        fil2 = open(path2 + route_ID + '.json', 'w')
                        fil2.write('{\n')
                        break
            else:
                prefix, event, value = next(parser)
                if prefix == "":
                    fil2.close()
                    break
                if event == 'start_map':
                    fil2.write('{\n')
                    comma_flag = False
                elif event == 'end_map':
                    fil2.write('}\n')
                elif event == 'map_key':
                    if comma_flag == False:
                        fil2.write('"' + value + '"' + ':')
                        comma_flag = True
                    else:
                        fil2.write(',"' + value + '"' + ':')
                elif event == 'number':
                    fil2.write(str(value))
                    # fil2.write(',')

    fil.close()


def process_zone_info_apply(BASE_DIR):
    path1 = path.join(BASE_DIR, 'data/model_apply_outputs/')

    with open(BASE_DIR + "/data/model_apply_inputs/new_route_data.json", 'rb') as file:
        route_data = json.load(file)
    route = list(route_data.keys())

    with open(BASE_DIR + '/data/model_apply_inputs/new_travel_times.json', 'rb') as file:
        travel_time_data = json.load(file)

    for i in range(len(route)):
        tra_time = travel_time_data[route[i]]
        with open(path1 + route[i] + '.json', 'w') as file:
            json.dump(tra_time, file)

        station_code = route_data[route[i]]['station_code']  # station code, e.g. DLA4
        zone_info = {}
        zone_info['station'] = route_data[route[i]]['station_code']
        zone_info['avg_tra_time'] = {}
        zone_info['avg_centroid'] = {}
        zone_info['depot_zone_tra_time_ratio'] = {}
        zone_info['zone_depot_tra_time_ratio'] = {}
        zone_info['P1P2'] = {}
        zone_dict = zone_set(route_data, route[i])
        zone_list = list(zone_dict.keys())
        station = get_station(route_data, route[i])
        for zone1 in zone_list:
            zone_info['avg_tra_time'][zone1] = {}
            zone_info['avg_centroid'][zone1] = {}
            zone_info['depot_zone_tra_time_ratio'][zone1] = {}
            zone_info['zone_depot_tra_time_ratio'][zone1] = {}
            zone_info['P1P2'][zone1] = {}
            for zone2 in zone_list:
                if zone1 == zone2:
                    zone_info['avg_tra_time'][zone1][zone2] = 0
                    zone_info['avg_centroid'][zone1][zone2] = 0
                    zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = 0
                    zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = 0
                    zone_info['P1P2'][zone1][zone2] = 0

                else:
                    zone_info['avg_tra_time'][zone1][zone2] = avg_travel_time(zone_dict[zone1],zone_dict[zone2],tra_time)
                    zone_info['avg_centroid'][zone1][zone2] = avg_tra_distance(zone_dict[zone1], zone_dict[zone2],
                                                                               route_data, route[i])
                    if zone1 == station_code or zone2 == station_code:
                        zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = 0
                        zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = 0
                    else:
                        zone_info['depot_zone_tra_time_ratio'][zone1][zone2] = depot_zone(zone_dict[zone2],station,tra_time)/depot_zone(zone_dict[zone1],station,tra_time)
                        zone_info['zone_depot_tra_time_ratio'][zone1][zone2] = zone_depot(zone_dict[zone2], station, tra_time) /zone_depot(zone_dict[zone1], station, tra_time)
                    result = check_main_zone(zone1,zone2,station_code)
                    zone_info['P1P2'][zone1][zone2] = result

        with open(path1 + route[i] + 'zone.json', 'w') as file:
            json.dump(zone_info, file)


