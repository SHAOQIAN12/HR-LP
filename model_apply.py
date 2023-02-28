from os import path
import os
from compute_seq import apply_model
from process_data import process_zone_info_apply
import json
import numpy as np
import score1
def load_json(file):
    with open(file, "rb") as f:
        output = json.load(f)
    return output

def cal_theta(data):
    t = []
    t.append(data['params']['t1'])
    t.append(data['params']['t2'])
    t.append(data['params']['t3'])
    t.append(data['params']['t4'])
    t.append(data['params']['t5'])
    t = np.array(t)
    return t


def read_json_data(filepath):
    try:
        with open(filepath, newline = '') as in_file:
            return json.load(in_file)
    except FileNotFoundError:
        print("The '{}' file is missing!".format(filepath))
    except json.JSONDecodeError:
        print("Error in the '{}' JSON data!".format(filepath))
    except Exception as e:
        print("Error when reading the '{}' file!".format(filepath))
        print(e)
    return None


if __name__ == '__main__':
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
    BASE_DIR = BASE_DIR + '/amz_v1'
    path1 = path.join(BASE_DIR, 'data/model_apply_inputs/')
    path2 = path.join(BASE_DIR, 'data/model_apply_outputs/')

    station_set = ['DLA3', 'DLA9', 'DLA7', 'DLA4', 'DLA8', 'DLA5', 'DSE4', 'DSE5', 'DSE2', 'DCH4', 'DCH3', 'DCH2',
                   'DCH1','DBO2', 'DBO3', 'DBO1', 'DAU1']

    print('Process apply data:')
    process_zone_info_apply(BASE_DIR)

    with open(path1 + "/new_route_data.json", 'rb') as fil:
        route_data = json.load(fil)

    routes = list(route_data.keys())
    k = 2
    print("Apply model")

    args_theta = {}
    for i in station_set:
        data = load_json(BASE_DIR + '/data/model_build_outputs/model_' + i + '.json')
        args_theta[i] = cal_theta(data)

    apply_model(routes, args_theta, route_data, k, path1, path2)

    print('Beginning Score Evaluation(test apply data... ', end='')

    # Read JSON time inputs
    model_build_time = read_json_data(os.path.join(BASE_DIR, 'data/model_score_timings/model_build_time.json'))
    model_apply_time = read_json_data(os.path.join(BASE_DIR, 'data/model_score_timings/model_apply_time.json'))

    output = score1.evaluate(
        actual_routes_json=os.path.join(BASE_DIR, 'data/model_score_inputs/new_actual_sequences.json'),
        invalid_scores_json=os.path.join(BASE_DIR, 'data/model_score_inputs/new_invalid_sequence_scores.json'),
        submission_json=os.path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json'),
        cost_matrices_json=os.path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json'),
        model_apply_time=model_apply_time.get("time"),
        model_build_time=model_build_time.get("time")
    )
    print('done')

    # Write Outputs to File
    output_path = os.path.join(BASE_DIR, 'data/model_score_outputs/scores.json')
    with open(output_path, 'w') as out_file:
        json.dump(output, out_file)

    # Print Pretty Output
    print("\nsubmission_score:", output.get('submission_score'))
    rt_show = output.get('route_scores')
    extra_str = None
    if len(rt_show.keys()) > 5:
        rt_show = dict(list(rt_show.items())[:5])
        extra_str = "..."
        print("\nFirst five route_scores:")
    else:
        print("\nAll route_scores:")
    for rt_key, rt_score in rt_show.items():
        print(rt_key, ": ", rt_score)
    if extra_str:
        print(extra_str)


