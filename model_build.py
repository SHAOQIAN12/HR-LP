# -*- coding: utf-8 -*-
from os import path
import os
import numpy as np
import json
from bayes_opt import BayesianOptimization, UtilityFunction
from math import ceil
from random import shuffle
from process_data import *
from compute_seq import compute_and_evaluate_routes



def bays_fun(t1, t2, t3, t4, t5):
    theta = np.array([t1, t2, t3, t4, t5])
    scores = compute_and_evaluate_routes(routes_station,theta,route_data_station,k,path3,path4)
    val_score = scores['submission_score']
    
    return -val_score

def split_routeID(routes):
    n = len(routes)
    #shuffle(routes)
    train = routes[:ceil(0.7 * n)]
    test = routes[ceil(0.7 * n):]
    return train, test

def process_dict(id,data):
    data_dict = {}
    for i in id:
        data_dict[i] =data[i]
    return data_dict

def load_json(file):
    with open(file, "rb") as f:
        output = json.load(f)
    return output

def dump_json(file, target):
    output_path = path.join(BASE_DIR, file)
    with open(output_path, 'w') as out_file:
        json.dump(target, out_file)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("there exists folder")


def split_data(index,route_id):
    train_id, test_id = split_routeID(route_id)
    print('train data num:',len(train_id))
    print('test data num:',len(test_id))
    build_route_data = process_dict(train_id, route_data)

    build_package_data = process_dict(train_id, package_data)

    build_actual_sequences = process_dict(train_id, actual_sequences)

    build_travel_times = process_dict(train_id, travel_times)

    build_invalid_sequence_scores = process_dict(train_id, invalid_sequence_scores)

    # output file
    dir = BASE_DIR+'/data/model_build_inputs_'+index
    mkdir(dir)

    dump_json(dir +'/route_data.json',build_route_data)
    dump_json(dir +'/package_data.json',build_package_data)
    dump_json(dir + '/actual_sequences.json', build_actual_sequences)
    dump_json(dir + '/travel_times.json', build_travel_times)
    dump_json(dir + '/invalid_sequence_scores.json', build_invalid_sequence_scores)


    dump_json(BASE_DIR + '/data/model_apply_inputs/new_route_data_' + index +'.json', process_dict(test_id, route_data))
    dump_json(BASE_DIR + '/data/model_apply_inputs/new_package_data_'+index+'.json',process_dict(test_id, package_data))
    dump_json(BASE_DIR + '/data/model_score_inputs/new_actual_sequences_'+index+'.json',process_dict(test_id, actual_sequences))
    dump_json(BASE_DIR + '/data/model_apply_inputs/new_travel_times_'+index+'.json',process_dict(test_id, travel_times))
    dump_json(BASE_DIR + '/data/model_score_inputs/new_invalid_sequence_scores_'+index+'.json',process_dict(test_id, invalid_sequence_scores))



if __name__ == "__main__":
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    
    station_set = ['DLA3', 'DLA9', 'DLA7', 'DLA4', 'DLA8', 'DLA5', 'DSE4', 'DSE5', 'DSE2', 'DCH4', 'DCH3', 'DCH2',
                   'DCH1','DBO2', 'DBO3', 'DBO1', 'DAU1']
    k = 2
    # 1. Split the data into 70%  for model build stage and 30% for model apply stage.
    route_data = load_json(BASE_DIR + "/data/model_build_inputs/route_data.json")
    package_data = load_json(BASE_DIR + "/data/model_build_inputs/package_data.json")
    actual_sequences = load_json(BASE_DIR + "/data/model_build_inputs/actual_sequences.json")
    travel_times = load_json(BASE_DIR + "/data/model_build_inputs/travel_times.json")
    invalid_sequence_scores = load_json(BASE_DIR+ "/data/model_build_inputs/invalid_sequence_scores.json")

    routeID = list(route_data.keys())
    high = []
    for i in routeID:
        if route_data[i]['route_score'] == 'High':
            high.append(i)

    station_dict = {}
    for i in station_set:
        station_dict[i] = []

    for i in high:
        for j in station_set:
            if route_data[i]['station_code'] == j:
                station_dict[j].append(i)

    for i in station_set:
        print('split', i, ':')
        split_data(i, station_dict[i])

    # 2. Combine model apply stage data

    path1 = path.join(BASE_DIR, 'data/model_apply_inputs/')
    path2 = path.join(BASE_DIR, 'data/model_score_inputs/')
    
    #
    # 2.1 combine new route data
    route_data = {}
    for i in station_set:
        data = load_json(path1 + "new_route_data_" + i + ".json")
        for j in data.keys():
            route_data[j] = data[j]

    dump_json('data/model_apply_inputs/new_route_data.json', route_data)

    # 2.2 combine new package data
    package_data = {}
    for i in station_set:
        data = load_json(path1 + "new_package_data_" + i + ".json")
        for j in data.keys():
            package_data[j] = data[j]

    dump_json('data/model_apply_inputs/new_package_data.json', package_data)

    # 2.3 combine new travel times data
    travel_time_data = {}
    for i in station_set:
        data = load_json(path1 + "new_travel_times_" + i + ".json")
        for j in data.keys():
            travel_time_data[j] = data[j]

    dump_json('data/model_apply_inputs/new_travel_times.json', travel_time_data)

    # 2.4 combine new actual_sequences data
    actual_sequences = {}
    for i in station_set:
        data = load_json(path2 + "new_actual_sequences_" + i + ".json")
        for j in data.keys():
            actual_sequences[j] = data[j]

    dump_json('data/model_score_inputs/new_actual_sequences.json', actual_sequences)

    # 2.5 combine new invalid sequences data
    invalid_sequence_scores = {}
    for i in station_set:
        data = load_json(path2 + "new_invalid_sequence_scores_" + i + ".json")
        for j in data.keys():
            invalid_sequence_scores[j] = data[j]

    dump_json('data/model_score_inputs/new_invalid_sequence_scores.json', invalid_sequence_scores)


    # 3. start training by depots
    for station in station_set:
        print('start training '+ station + ' parameter:')
        path3 = path.join(BASE_DIR, 'data/model_build_inputs_' + station + '/')
        path4 = path.join(BASE_DIR, 'data/model_build_outputs/')

        process_zone_info(BASE_DIR, station)

        n_initial_points = 20

        with open(path3 + "route_data.json", 'rb') as fil:
            route_data_station = json.load(fil)

        routes_station = list(route_data_station.keys())

        bounds = {'t1': [1.0, 10.0], 't2': [1.0, 10.0],'t3': [1.0, 10.0], 't4': [1.0, 10.0],'t5': [1.0, 10.0]}

        optimizer = BayesianOptimization(f=None,pbounds=bounds,verbose=2,random_state=42)

        utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

        params1 = {'t1': 10.0, 't2': 1.0,'t3': 1.0,'t4': 1.0,'t5': 1.0}

        obj_init = bays_fun(**params1)
        optimizer.register(params=params1, target=obj_init)
        print(params1, obj_init)

        dump_json('data/model_build_outputs/' + 'model_' +station + '.json',optimizer.max)

        for i in range(1, 21):
            print('Iteration:',i)
            sampled_point = optimizer.space.random_sample()
            next_point = optimizer.space.array_to_params(sampled_point)
            obj = bays_fun(**next_point)
            optimizer.register(params=next_point, target=obj)
            print(obj, next_point)

            dump_json('data/model_build_outputs/' + 'model_' +station + '.json',optimizer.max)

        for i in range(21, 101):
            print('Iteration:', i)
            next_point = optimizer.suggest(utility)
            obj = bays_fun(**next_point)
            optimizer.register(params=next_point, target=obj)
            print(obj, next_point)
            dump_json('data/model_build_outputs/' + 'model_' + station + '.json', optimizer.max)





