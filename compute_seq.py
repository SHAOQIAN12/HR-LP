# -*- coding: utf-8 -*-
import os
from multiprocessing import Pool
from score import *
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
from process_data import get_key, zone_set


bigM = 99999999  # a very large number


def identify_stations(route_data):
    rotas_id = list(route_data.keys())
    stations_dict = {}
    for r_id in rotas_id:
        rota_dict = route_data[r_id]

        for stop in rota_dict['stops']:
            if rota_dict['stops'][stop]['type'] == 'Station':
                station = stop
                break

        stations_dict[r_id] = station

    return stations_dict


def normalization(dict_info):
    value = []
    for i in dict_info.keys():
        for j in dict_info[i].keys():
            value.append(dict_info[i][j])
    value = np.array(value)
    MAX = value.max()
    MIN = value.min()
    new = {}
    for i in dict_info.keys():
        new[i] = {}
        for j in dict_info[i].keys():
            new[i][j] = round((dict_info[i][j] - MIN) / (MAX - MIN), 4)
    return new


def create_data_model(routeID, path2, theta):
    with open(path2 + routeID + "zone.json", 'rb') as fil:
        zone_info = json.load(fil)
    data = {}
    station = zone_info['station']
    time_distance = []
    index = 0
    zone_id = {}
    avg_tra_time = normalization(zone_info['avg_tra_time'])
    avg_centroid = normalization(zone_info['avg_centroid'])
    depot_zone_tra_time_ratio = normalization(zone_info['depot_zone_tra_time_ratio'])
    zone_depot_tra_time_ratio = normalization(zone_info['zone_depot_tra_time_ratio'])
    p1p2 = zone_info['P1P2']
    for i in zone_info['avg_tra_time'].keys():
        time_distance1 = []
        zone_id[i] = index
        if i == station:
            station_index = index
        index += 1
        for j in zone_info['avg_tra_time'].keys():
            time_distance1.append(int(1000 * (
                    avg_tra_time[i][j] * theta[0] + avg_centroid[i][j] * theta[1] +
                    depot_zone_tra_time_ratio[i][j] * theta[2] + zone_depot_tra_time_ratio[i][j] * theta[3] + p1p2[i][
                        j] * theta[4])))
        time_distance.append(time_distance1)
    data['distance_matrix'] = time_distance
    data['num_vehicles'] = 1
    data['depot'] = station_index
    return data, zone_id


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def compute_zone_seq(args):
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    routeID, theta, dataRoute, k, path1, path2 = args
    sequence = {}
    zone_id = {}
    data, zone_id = create_data_model(routeID, path2, theta)

    manager = pywrapcp.RoutingIndexManager(len((data['distance_matrix'])),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # using first_solution_strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)

    solution = routing.SolveWithParameters(search_parameters)
    routes = get_routes(solution, routing, manager)

    R = []
    for i, route in enumerate(routes):
        R = route
    for j in zone_id.keys():
        sequence[j] = R.index(zone_id[j])
    return sequence, zone_id


def get_keys(d, value):
    return [k for k, v in d.items() if v == value]


def find_nearest_stop(zone1, zone2, k, route_data, route):
    distance = {}
    for i in zone1:
        dist = 0
        vector1 = np.array([route_data[route]['stops'][i]['lat'], route_data[route]['stops'][i]['lng']])
        for j in zone2:
            vector2 = np.array([route_data[route]['stops'][j]['lat'], route_data[route]['stops'][j]['lng']])
            dist += np.linalg.norm(vector1 - vector2, ord=1)
        distance[i] = round(dist / len(zone2), 4)
    m = sorted(distance.items(), key=lambda kv: (kv[1], kv[0]))

    stop = []
    if len(zone1) < k:
        k = len(zone1)
    for i in range(k):
        stop.append(m[i][0])
    return stop


def create_stop_data_model(zone, tra_time, initial, end):
    """Stores the data for the problem."""
    data = {}
    time_distance = []
    zone1 = []
    zone1.append(initial)
    for i in zone:
        if i != initial and i != end:
            zone1.append(i)
    zone1.append(end)
    for i in zone1:
        time_distance1 = []
        for j in zone1:
            time_distance1.append(int(1000 * tra_time[i][j]))
        time_distance.append(time_distance1)

    data['distance_matrix'] = time_distance
    data['num_vehicles'] = 1
    data['starts'] = [0]
    data['ends'] = [len(zone1) - 1]
    stop_id = {}
    for i in range(len(zone1)):
        stop_id[zone1[i]] = i
    return data, stop_id


def cal_stop_seq(zone, tra_time, initial, end):
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    data, stop_id = create_stop_data_model(zone, tra_time, initial, end)

    manager = pywrapcp.RoutingIndexManager(len((data['distance_matrix'])),
                                           data['num_vehicles'], data['starts'], data['ends'])
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # using first_solution_strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS)

    solution = routing.SolveWithParameters(search_parameters)
    routes = get_routes(solution, routing, manager)

    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

    R = []
    for i, route in enumerate(routes):
        R = route
    best_route = []
    for i in R:
        best_route.append(get_key(stop_id, i)[0])
    return route_distance, best_route


def compute_stop_seq(zone1, zone2, zone3, k, dataRoute, route, tra_time):
    zone1 = list(zone1)
    zone2 = list(zone2)
    zone3 = list(zone3)
    initial_stops = find_nearest_stop(zone2, zone1, k, dataRoute, route)
    end_stops = find_nearest_stop(zone2, zone3, k, dataRoute, route)
    result = 999999
    best = []
    for i in initial_stops:
        for j in end_stops:
            if i == j: continue
            best_distance, best_route = cal_stop_seq(zone2, tra_time, i, j)
            if best_distance < result:
                result = best_distance
                best = best_route
    return best


def compute_route(args):
    routeID, theta, dataRoute, k, path1, path2 = args
    zone_dict = zone_set(dataRoute, routeID)
    t1 = time.time()
    seq, zone_id = compute_zone_seq(args)
    t2 = time.time() - t1
    zone_seq_list = sorted(seq.items(), key=lambda kv: (kv[1], kv[0]))
    zone_seq = []
    for i in range(len(zone_seq_list)):
        zone_seq.append(zone_seq_list[i][0])

    with open(path2 + routeID + ".json", 'rb') as fil:
        tra_time = json.load(fil)
    stop_seq = zone_dict[zone_seq[0]]
    tra_time = normalization(tra_time)
    length = len(zone_seq)
    time_record = {}
    time_record[routeID] = {}
    for i in range(length - 2):
        time_record[routeID][i] = {}
        t1 = time.time()
        if len(zone_dict[zone_seq[i + 1]]) == 1:
            best = zone_dict[zone_seq[i + 1]]
        elif len(zone_dict[zone_seq[i + 1]]) == 2:
            stop = np.array(zone_dict[zone_seq[i + 1]])
            tra_1 = tra_time[stop_seq[-1]][stop[0]] + tra_time[stop[0]][stop[1]]
            tra_2 = tra_time[stop_seq[-1]][stop[1]] + tra_time[stop[1]][stop[0]]
            if tra_1 < tra_2:
                best = [stop[0], stop[1]]

            else:
                best = [stop[1], stop[0]]
                # print(tra_2, best)
        else:
            best = compute_stop_seq(zone_dict[zone_seq[i]], zone_dict[zone_seq[i + 1]], zone_dict[zone_seq[i + 2]], k,
                                    dataRoute, routeID, tra_time)
        stop_seq += best
        time_record[routeID][i]['stops'] = len(best)
        time_record[routeID][i]['time'] = time.time() - t1

    zone_last = zone_dict[zone_seq[length - 1]]

    t1 = time.time()
    time_record[routeID][length - 1] = {}
    if len(zone_last) == 1:
        best = zone_last
    elif len(zone_last) == 2:
        stop = zone_last
        tra_1 = tra_time[stop_seq[-1]][stop[0]] + tra_time[stop[0]][stop[1]]
        tra_2 = tra_time[stop_seq[-1]][stop[1]] + tra_time[stop[1]][stop[0]]
        if tra_1 < tra_2:
            best = [stop[0], stop[1]]
        else:
            best = [stop[1], stop[0]]
    else:
        best = compute_stop_seq(zone_dict[zone_seq[length - 2]], zone_dict[zone_seq[length - 1]],
                                zone_dict[zone_seq[0]], k, dataRoute, routeID, tra_time)

    stop_seq += best
    time_record[routeID][length - 1]['stops'] = len(best)
    time_record[routeID][length - 1]['time'] = time.time() - t1
    time_record[routeID]['zone time'] = t2

    return (routeID, stop_seq)


def gen_proposed_route_json(proposed_routes):
    pro_routes = {}

    for route in proposed_routes:
        pro_routes[route] = {}
        pro_routes[route]['proposed'] = {}

        k = 0
        for stop in proposed_routes[route]:
            pro_routes[route]['proposed'][stop] = k
            k += 1

    return pro_routes



def compute_all_routes_build(route_ids, theta, Route_data, k, path1, path2):
    args = []
    for routeID in route_ids:
        args.append([routeID, theta, Route_data, k, path1, path2])

    with Pool() as p:
        routes = p.map(compute_route, args)

    routes = dict(routes)

    route_f = gen_proposed_route_json(routes)

    return route_f


# apply stage
def compute_all_routes(route_ids, args_theta, Route_data, k, path1, path2):

    args = []
    for routeID in route_ids:
        for station in args_theta.keys():
            if Route_data[routeID]['station_code'] == station:
                args.append([routeID, args_theta[station], Route_data, k, path1, path2])
    with Pool() as p:
        routes = p.map(compute_route, args)  # compute_route compute route sequences.

    routes = dict(routes)

    route_f = gen_proposed_route_json(routes)

    return route_f



def my_evaluate(actual_routes_json, submission_json, invalid_scores_json, path, **kwargs):
    actual_routes = read_json_data(actual_routes_json)
    good_format(actual_routes, 'actual', actual_routes_json)
    submission = read_json_data(submission_json)
    good_format(submission, 'proposed', submission_json)
    invalid_scores = read_json_data(invalid_scores_json)
    good_format(invalid_scores, 'invalids', invalid_scores_json)
    scores = {'submission_score': 'x', 'route_scores': {}, 'route_feasibility': {}}
    for kwarg in kwargs:
        scores[kwarg] = kwargs[kwarg]
    k = 1
    for route in actual_routes:
        print("Evaluating route ", k)
        if route not in submission:
            scores['route_scores'][route] = invalid_scores[route]
            scores['route_feasibility'][route] = False
        else:
            actual_dict = actual_routes[route]
            actual = route2list(actual_dict)
            try:
                sub_dict = submission[route]
                sub = route2list(sub_dict)
            except:
                scores['route_scores'][route] = invalid_scores[route]
                scores['route_feasibility'][route] = False
            else:
                if isinvalid(actual, sub):
                    scores['route_scores'][route] = invalid_scores[route]
                    scores['route_feasibility'][route] = False
                else:
                    with open(path + route + '.json') as fil2:
                        cost_mat = json.load(fil2)

                        # cost_mat=cost_matrices[route]
                    scores['route_scores'][route] = score(actual, sub, cost_mat)
                    scores['route_feasibility'][route] = True
        k += 1
    submission_score = np.mean(list(scores['route_scores'].values()))
    scores['submission_score'] = submission_score
    return scores


def parallel_evaluate_scores(actual_routes_json,submission_json,invalid_scores_json,
                         path):
    actual_routes = read_json_data(actual_routes_json)
    good_format(actual_routes, 'actual', actual_routes_json)
    submission = read_json_data(submission_json)
    good_format(submission, 'proposed', submission_json)
    invalid_scores = read_json_data(invalid_scores_json)
    good_format(invalid_scores, 'invalids', invalid_scores_json)
    args = []
    for route in submission:
        args.append([route, actual_routes, submission, path])

    with Pool() as p:
        scores_list = p.map(evaluate_single_route, args)
    return scores_list


def evaluate_single_route(args):
    route, actual_routes, submission, path = args

    scores = {'submission_score': 'x', 'route_scores': {}, 'route_feasibility': {}}

    if route not in submission:
        scores['route_scores'][route] = invalid_scores[route]
        scores['route_feasibility'][route] = False
    else:
        actual_dict = actual_routes[route]
        actual = route2list(actual_dict)
        try:
            sub_dict = submission[route]
            sub = route2list(sub_dict)
        except:
            scores['route_scores'][route] = invalid_scores[route]
            scores['route_feasibility'][route] = False
        else:
            if isinvalid(actual, sub):
                scores['route_scores'][route] = invalid_scores[route]
                scores['route_feasibility'][route] = False
            else:
                with open(path + route + '.json') as fil2:
                    cost_mat = json.load(fil2)

                actual = tuple(actual)
                sub = tuple(sub)

                scores['route_scores'][route] = score(actual, sub, cost_mat)
                scores['route_feasibility'][route] = True

    return scores


def cal_scores(scores_list):
    scores = {'submission_score': 'x', 'route_scores': {}, 'route_feasibility': {}}

    for dic in scores_list:
        keys = list(dic['route_scores'].keys())
        values1 = list(dic['route_scores'].values())
        values2 = list(dic['route_feasibility'].values())
        route = keys[0]
        route_score = values1[0]
        route_feas = values2[0]
        scores['route_scores'][route] = route_score
        scores['route_feasibility'][route] = route_feas

    submission_score = np.mean(list(scores['route_scores'].values()))
    scores['submission_score'] = submission_score

    return scores


def compute_and_evaluate_routes(route_ids, theta, dataRoute, k, path1, path2):
    routes_final = compute_all_routes_build(route_ids, theta, dataRoute, k, path1, path2)

    with open(path2 + "submission.json", 'w') as fil:
        json.dump(routes_final, fil)

    scores_list = parallel_evaluate_scores(path1 + 'actual_sequences.json', path2 + 'submission.json',
                                       path1 + 'invalid_sequence_scores.json',path2)

    scores = cal_scores(scores_list)

    return scores


def apply_model(route_ids, args_theta, Route_data, k, path1, path2):
    routes_final = compute_all_routes(route_ids, args_theta, Route_data, k, path1, path2)

    with open(path2 + "proposed_sequences.json", 'w') as file:
        json.dump(routes_final, file)
    for i in route_ids:
        os.remove(path2 + str(i) + '.json')
        os.remove(path2 + str(i) + 'zone.json')



def closestNode(route, node, route_data):
    x_node = route_data[route].get('stops').get(node).get('lat')
    y_node = route_data[route].get('stops').get(node).get('lng')
    min_node = bigM
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
