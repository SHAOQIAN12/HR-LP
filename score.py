import numpy as np
import json
import sys


def read_json_data(filepath):
    try:
        with open(filepath, newline = '') as in_file:
            file=json.load(in_file)
            in_file.close()
    except FileNotFoundError:
        print("The '{}' file is missing!".format(filepath))
        sys.exit()
    except Exception as e:
        print("Error when reading the '{}' file!".format(filepath))
        print(e)
        sys.exit()
    return file


def good_format(file,input_type,filepath):
    for route in file:
        if route[:8]!='RouteID_':
            raise JSONDecodeError('Improper route ID in {}. Every route must be denoted by a string that begins with "RouteID_".'.format(filepath))
    if input_type=='proposed' or input_type=='actual':
        for route in file:
            if type(file[route])!=dict or len(file[route])!=1: 
                raise JSONDecodeError('Improper route in {}. Each route ID must map to a dictionary with a single key.'.format(filepath))
            if input_type not in file[route]:
                if input_type=='proposed':
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in a proposed sequence file must have the key, "proposed".'.format(filepath))
                else:
                    raise JSONDecodeError('Improper route in {}. Each route\'s dictionary in an actual sequence file must have the key, "actual".'.format(filepath))
            if type(file[route][input_type])!=dict:
                raise JSONDecodeError('Improper route in {}. Each sequence must be in the form of a dictionary.'.format(filepath))
            num_stops=len(file[route][input_type])
            for stop in file[route][input_type]:
                if type(stop)!=str or len(stop)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                stop_num=file[route][input_type][stop]
                if type(stop_num)!=int or stop_num>=num_stops:
                    raise JSONDecodeError('Improper stop number in {}. Each stop\'s position number, x, must be an integer in the range 0<=x<N where N is the number of stops in the route (including the depot).'.format(filepath))
    if input_type=='costs':
        for route in file:
            if type(file[route])!=dict:
                raise JSONDecodeError('Improper matrix in {}. Each cost matrix must be a dictionary.'.format(filepath)) 
            for origin in file[route]:
                if type(origin)!=str or len(origin)!=2:
                    raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                if type(file[route][origin])!=dict:
                    raise JSONDecodeError('Improper matrix in {}. Each origin in a cost matrix must map to a dictionary of destinations'.format(filepath))
                for dest in file[route][origin]:
                    if type(dest)!=str or len(dest)!=2:
                        raise JSONDecodeError('Improper stop ID in {}. Each stop must be denoted by a two-letter ID string.'.format(filepath))
                    if not(type(file[route][origin][dest])==float or type(file[route][origin][dest])==int):
                        raise JSONDecodeError('Improper time in {}. Every travel time must be a float or int.'.format(filepath))
    if input_type=='invalids':
        for route in file:
            if not(type(file[route])==float or type(file[route])==int):
                raise JSONDecodeError('Improper score in {}. Every score in an invalid score file must be a float or int.'.format(filepath))

class JSONDecodeError(Exception):
    pass

def evaluate(actual_routes_json,submission_json,cost_matrices_json, invalid_scores_json,**kwargs):
    actual_routes=read_json_data(actual_routes_json)
    good_format(actual_routes,'actual',actual_routes_json)
    submission=read_json_data(submission_json)
    good_format(submission,'proposed',submission_json)
    cost_matrices=read_json_data(cost_matrices_json)
    good_format(cost_matrices,'costs',cost_matrices_json)
    invalid_scores=read_json_data(invalid_scores_json)
    good_format(invalid_scores,'invalids',invalid_scores_json)
    scores={'submission_score':'x','route_scores':{},'route_feasibility':{}}
    for kwarg in kwargs:
        scores[kwarg]=kwargs[kwarg]
    for route in actual_routes:
        if route not in submission:
            scores['route_scores'][route]=invalid_scores[route]
            scores['route_feasibility'][route]=False
        else:
            actual_dict=actual_routes[route]
            actual=route2list(actual_dict)
            try:
                sub_dict=submission[route]
                sub=route2list(sub_dict)
            except:
                scores['route_scores'][route]=invalid_scores[route]
                scores['route_feasibility'][route]=False
            else:
                if isinvalid(actual,sub):
                    scores['route_scores'][route]=invalid_scores[route]
                    scores['route_feasibility'][route]=False
                else:
                     cost_mat=cost_matrices[route]
                     scores['route_scores'][route]=score(actual,sub,cost_mat)
                     scores['route_feasibility'][route]=True
    submission_score=np.mean(list(scores['route_scores'].values()))
    scores['submission_score']=submission_score
    return scores


def score(actual,sub,cost_mat,g=1000):
    norm_mat=normalize_matrix(cost_mat)
    return seq_dev(actual,sub)*erp_per_edit(actual,sub,norm_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count


def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):

    if memo==None:
        memo={}

    actual_tuple=actual
    sub_tuple=sub
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    new_mat=mat.copy()
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat


def gap_sum(path,g):
    res=0
    if len(path) == 0:
        return res
    for p in path:
        res+=g
    return res
    

def dist_erp(p_1,p_2,mat,g=1000):

    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist


def seq_dev(actual,sub):
    actual=actual[1:-1]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum


def isinvalid(actual,sub):
    if len(actual)!=len(sub) or set(actual)!=set(sub):
        return True
    elif actual[0]!=sub[0]:
        return True
    else:
        return False

def route2list(route_dict):

    if 'proposed' in route_dict:
        stops=route_dict['proposed']
    elif 'actual' in route_dict:
        stops=route_dict['actual']
    route_list=[0]*(len(stops)+1)
    for stop in stops:
        route_list[stops[stop]]=stop
    route_list[-1]=route_list[0]
    return route_list

