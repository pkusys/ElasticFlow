from mip import *
import numpy as np
import copy
import math


def compute_maximum_lease(expect_maximum_end_time, lease_time_interval, cur_lease_index):
    return math.floor(expect_maximum_end_time / lease_time_interval) - cur_lease_index
    # return math.ceil(expect_maximum_end_time / lease_time_interval) - cur_lease_index


def compute_emergence(expected_remaining_time, true_remaining_time, required_gpu_num):
    # return -1 * true_remaining_time * required_gpu_num
    # return -expected_remaining_time * required_gpu_num
    # return 1 * true_remaining_time * required_gpu_num
    return -1 * true_remaining_time * required_gpu_num

class NoPreemptMIPSolver(object):
    def __init__(self, method):
        self.method = method
    

    def batch_fast_check_if_packable(self, required_resource_list, required_block_list, maximum_block_list, existing_solution, resource_num_list):
        if len(maximum_block_list) == 0:
            return True, copy.deepcopy(existing_solution)
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        
        resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
        # resource_tuple_list.sort(key=lambda e: (e[0], e[1], -e[2]))
        # resource_tuple_list.sort(key=lambda e: (e[1], e[0], e[2]))
        resource_tuple_list.sort(key=lambda e: (e[0]-e[1], e[1], e[2]))

        for maximum_block, required_block, required_resource in resource_tuple_list:
            if True:
                feasible, cnt = False, 0
                # for i in range(maximum_block):
                for i in range(maximum_block - 1, -1, -1):
                    if solution[i] >= required_resource:
                        cnt += 1
                        solution[i] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            if False:
                feasible = False
                index_block = -1
                for i in range(maximum_block - 1, required_block, -1):
                    doable = True
                    for j in range(i, i - required_block - 1, -1):
                        if solution[j] < required_resource:
                            doable = False
                    if doable:
                        index_block = i
                        break

                if index_block == -1: break
                feasible = True
                for i in range(index_block, index_block - required_block - 1, -1):
                    solution[i] -= required_resource
        return feasible, solution
    

    def batch_fast_job_selection(self, required_resource_list, required_block_list, maximum_block_list, existing_solution, resource_num_list):
        if len(maximum_block_list) == 0:
            return True, list()
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])

        solution_matrix = [i for i in range(len(maximum_block_list))]
        resource_tuple_list = [(a, b, c, i) for a, b, c, i in zip(maximum_block_list, required_block_list, required_resource_list, solution_matrix)]
        # resource_tuple_list.sort(key=lambda e: (e[1], e[0], e[2]))
        # resource_tuple_list.sort(key=lambda e: (e[0], e[1], -e[2]))
        resource_tuple_list.sort(key=lambda e: (e[0]-e[1], e[1], e[2]))
        
        for maximum_block, required_block, required_resource, idx in resource_tuple_list:
            feasible, cnt = False, 0
            cache_solution = [0 for _ in range(maximum_block)]
            # for i in range(maximum_block - 1, -1, -1):
            if True:
                for i in range(maximum_block - 1, -1, -1):
                # for i in range(0, maximum_block):
                    if solution[i] >= required_resource:
                        cnt += 1
                        solution[i] -= required_resource
                        cache_solution[i] = 1
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            if False:
                index_block = -1
                for i in range(maximum_block - 1, required_block, -1):
                    feasible = True
                    for j in range(i, i - required_block - 1, -1):
                        if solution[j] < required_resource:
                            feasible = False
                    if feasible:
                        index_block = i
                        break
                if index_block == -1: 
                    feasible = False
                    break
                for i in range(index_block, index_block - required_block - 1, -1):
                    solution[i] -= required_resource
                    cache_solution[i] = 1
                    
            solution_matrix[idx] = cache_solution
            feasible = True
                
        return feasible, solution_matrix



class MIPSolver(object):
    def __init__(self, method):
        self.method = method
    
    def check_if_packable(self, required_resource_list, required_block_list, maximum_block_list, resource_num_list, method):
        if method == 'knapsack':
            maximum_block = max(maximum_block_list)
            m = Model(solver_name=GRB)
            var_len = len(required_resource_list) * maximum_block
            X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
            m.objective = maximize(X[-1])
            # job-wise
            for i in range(len(required_resource_list)):
                m += xsum(X[j] for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) == required_block_list[i]
                if maximum_block_list[i] <  maximum_block:
                    m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

            # resource-wise
            for i in range(maximum_block):
                m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i]
            m.optimize()
            
            feasible = not any([X[i].x is None for i in range(var_len)])
            solution = list()
            if feasible:
                for block_idx in range(maximum_block):
                    left_resource_num = resource_num_list[resource_num_list]
                    for var_idx in range(block_idx, var_len, maximum_block):
                        left_resource_num -= X[var_idx].x * required_resource_list[var_idx // maximum_block]
                    solution.append(left_resource_num)
        elif method == 'greedy':
            maximum_block = max(maximum_block_list)
            solution = [resource_num_list[i] for i in range(maximum_block)]
            resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
            resource_tuple_list.sort(reverse=True)
            for maximum_block, required_block, required_resource in resource_tuple_list:
                feasible, cnt = False, 0
                for i in range(maximum_block - 1, -1, -1):
                    if solution[i] >= required_resource:
                        cnt += 1
                        solution[i] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            
        elif method == 'greedy-smooth':
            maximum_block = max(maximum_block_list)
            solution = [resource_num_list[i] for i in range(maximum_block)]
            resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
            resource_tuple_list.sort(key=lambda e: (e[0] - e[1], -e[0], -e[2]))
            for maximum_block, required_block, required_resource in resource_tuple_list:
                feasible, cnt = False, 0
                # idx_resource_pair = sorted([(idx, resource) for idx, resource in enumerate(solution)], key=lambda e: (e[1], e[0]), reverse=True)
                idx_resource_pair = sorted([(idx, resource) for idx, resource in enumerate(solution)], key=lambda e: (e[0], e[1]), reverse=True)

                for idx, resource in idx_resource_pair:
                    if idx < maximum_block and resource >= required_resource:
                        cnt += 1
                        solution[idx] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            
        return feasible, solution
    

    def fast_check_if_packable(self, required_resource, required_block, maximum_block, existing_solution, resource_num_list):
        if required_block > maximum_block: return False, existing_solution
        feasible_block = 0
        if len(existing_solution) < maximum_block:
            existing_solution += [resource_num_list[i] for i in range(len(existing_solution), maximum_block)]

        for i in range(maximum_block):
            if existing_solution[i] >= required_resource:
                feasible_block += 1
        
        feasible = feasible_block >= required_block
        if feasible:
            for i in range(maximum_block - 1, -1, -1):
                if existing_solution[i] >= required_resource and required_block > 0:
                    existing_solution[i] -= required_resource
                    required_block -= 1
                
        return feasible, existing_solution

    def fast_job_cache_solution(self, required_resource_list, required_block_list, maximum_block_list, 
                                            existing_solution, resource_num_list):
        if len(maximum_block_list) == 0:
            return True, copy.deepcopy(existing_solution), None
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        required_resource, required_block, maximum_block = required_resource_list[-1], required_block_list[-1], maximum_block_list[-1]
        # fast job cache solution 
        cache_solution = [0 for _ in range(maximum_block)]
        feasible, cnt = False, 0
        for i in range(maximum_block - 1, -1, -1):
            if solution[i] >= required_resource:
                cnt += 1
                solution[i] -= required_resource
                cache_solution[i] = 1
            if cnt == required_block:
                feasible = True
                break
        if not feasible:
            return feasible, None, None



        required_resource_list, required_block_list, maximum_block_list = required_resource_list[:-1], required_block_list[:-1], maximum_block_list[:-1]
        resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
        resource_tuple_list.sort(key=lambda e: (-e[2], e[0] - e[1], -e[0]))

        for maximum_block, required_block, required_resource in resource_tuple_list:
            feasible, cnt = False, 0
            for i in range(maximum_block - 1, -1, -1):
                if solution[i] >= required_resource:
                    cnt += 1
                    solution[i] -= required_resource
                if cnt == required_block:
                    feasible = True
                    break
            if not feasible: break
                
        return feasible, solution, cache_solution


    def batch_fast_check_if_packable(self, soft_list, value_list, soft_id_list, required_resource_list, required_block_list, maximum_block_list, in_block_list, existing_solution, resource_num_list):

        if len(maximum_block_list) == 0:
            return True, copy.deepcopy(existing_solution)
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        
        resource_tuple_list = [(a, b, c, job) for a, b, c, job in zip(maximum_block_list, required_block_list, required_resource_list, in_block_list)]
        resource_tuple_list.sort(key=lambda e: (-e[2], e[0] - e[1], -e[0]))
        visit_list = list()
        for maximum_block, required_block, required_resource, job in resource_tuple_list:
            if job['job_id'] not in visit_list:
                visit_list.append(job['job_id'])
            else:
                continue 
            feasible, cnt = False, 0
            for i in range(maximum_block - 1, -1, -1):
                if solution[i] >= required_resource:
                    cnt += 1
                    solution[i] -= required_resource
                if cnt == required_block:
                    feasible = True
                    break
            if not feasible: break
            
        return feasible, solution
    



    def batch_fast_job_selection(self, soft_list, value_list, soft_id_list, required_resource_list, required_block_list, maximum_block_list, in_block_list, existing_solution, resource_num_list):
        if len(maximum_block_list) == 0:
            return True, list(), list()
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])

        solution_matrix = [i for i in range(len(maximum_block_list))]
        soft_matrix = [0 for i in range(len(maximum_block_list))]
        resource_tuple_list = [(a, b, c, i, soft_id, value, job) for a, b, c, i, soft_id, value, job in zip(maximum_block_list, required_block_list, required_resource_list, solution_matrix, soft_id_list, value_list, in_block_list)]
        resource_tuple_list.sort(key=lambda e: (-e[-2], -e[2], e[0] - e[1], -e[0]))
        
        visit_list = list()
        for maximum_block, required_block, required_resource, idx, soft_id, value, job in resource_tuple_list:
            if job['job_id'] in visit_list:
                continue 

            feasible, cnt = False, 0
            cache_solution = [0 for _ in range(maximum_block)]
            for i in range(maximum_block - 1, -1, -1):
                if solution[i] >= required_resource:
                    cnt += 1
                    solution[i] -= required_resource
                    cache_solution[i] = 1
                if cnt == required_block:
                    feasible = True
                    break
            if feasible:
                solution_matrix[idx] = cache_solution
                soft_matrix[idx] = 1
                visit_list.append(job['job_id'])
            if not feasible: 
                for i in range(maximum_block - 1, -1, -1):
                    if cache_solution[i] == 1:
                        solution[i] += required_resource
                        
        for job in in_block_list:
            if job['job_id'] not in visit_list:
                import pdb; pdb.set_trace()
                return False, list(), list() 
        
        return feasible, solution_matrix, soft_matrix
    

    # @timeout_decorator.timeout()
    def job_selection(self, soft_list, value_list, soft_id_list, required_resource_list, required_block_list, maximum_block_list, resource_num_list, objective, max_seconds=5):
        # info = {
        #     'required_resource_list': required_resource_list,
        #     'required_block_list' : required_block_list,
        #     'maximum_block_list': maximum_block_list,
        #     'resource_num_list': resource_num_list,
        #     'objective': objective
        # }
        
        # np.save("info.npy", info)
        max_resource_num = 1.0 * max(resource_num_list)
        maximum_block = max(maximum_block_list)
        if isinstance(resource_num_list, int):
            resource_num_list = [resource_num_list for _ in range(maximum_block)]
        m = Model(solver_name=GRB)
        var_len = len(required_resource_list) * maximum_block
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        S = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list))]

        obj_list = [S[j] * value_list[j] for j in range(len(required_resource_list))]
        if objective == 'random':
            obj_list = [S[0] * value_list[0]]
            # pass
            # m.objective = maximize(X[-1])
        elif objective == 'minimize':
            for j in range(0, var_len, maximum_block):
                obj_list.append(-X[j] * required_resource_list[j // maximum_block] / max_resource_num)
        elif objective == 'maximize':
            for j in range(0, var_len, maximum_block):
                obj_list.append(X[j] * required_resource_list[j // maximum_block] / max_resource_num)
            # m.objective = maximize(xsum(X[j] * required_resource_list[j // maximum_block]  for j in range(0, var_len, maximum_block)) )
            # m.objective = maximize(xsum(X[j]  for j in range(0, var_len, maximum_block)) )
        else:
            raise NotImplementedError
        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))
        # job-wise
        for i in range(len(required_resource_list)):
            m += xsum(X[j] for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) == S[i] * required_block_list[i]
            if maximum_block_list[i] <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

        # resource-wise
        for i in range(maximum_block):
            # m += xsum(X[j] * required_resource_list[j // maximum_block] * S[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] * S[j // maximum_block]
            m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i] #  * S[j // maximum_block]

        i = 0
        while i < len(required_resource_list):
            if soft_id_list[i] != 0:
                i += 1
                continue 
            left = i
            right = len(required_resource_list)
            for j in range(i+1, len(required_resource_list)):
                if soft_id_list[j] == 0:
                    right = j
                    break
            m += xsum(S[j] for j in range(left, right)) == 1
            i = right

        m.optimize(max_seconds=max_seconds, max_seconds_same_incumbent=1)
        solution_matrix = list()
        

        for i in range(len(required_resource_list)):
            start = i * maximum_block
            solution = list()
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                solution.append(res)
            solution_matrix.append(solution)
        soft_matrix = list()
        for i in range(len(soft_id_list)):
            soft_matrix.append(S[i].x)
        return solution_matrix, soft_matrix




class MIPSolverResourceUserConstrain(object):
    def __init__(self, method):
        self.method = method
    
    def check_if_packable(self, required_resource_list, required_block_list, maximum_block_list, resource_num_list, method):
        if method == 'knapsack':
            maximum_block = max(maximum_block_list)
            m = Model(solver_name=GRB)
            var_len = len(required_resource_list) * maximum_block
            X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
            m.objective = maximize(X[-1])
            # job-wise
            for i in range(len(required_resource_list)):
                m += xsum(X[j] for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) == required_block_list[i]
                if maximum_block_list[i] <  maximum_block:
                    m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

            # resource-wise
            for i in range(maximum_block):
                m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(i, var_len, maximum_block) ) <= resource_num_list[i]
            m.optimize()
            
            feasible = not any([X[i].x is None for i in range(var_len)])
            solution = list()
            if feasible:
                for block_idx in range(maximum_block):
                    left_resource_num = resource_num_list[resource_num_list]
                    for var_idx in range(block_idx, var_len, maximum_block):
                        left_resource_num -= X[var_idx].x * required_resource_list[var_idx // maximum_block]
                    solution.append(left_resource_num)
        elif method == 'greedy':
            maximum_block = max(maximum_block_list)
            solution = [resource_num_list[i] for i in range(maximum_block)]
            resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
            resource_tuple_list.sort(reverse=True)
            for maximum_block, required_block, required_resource in resource_tuple_list:
                feasible, cnt = False, 0
                for i in range(maximum_block - 1, -1, -1):
                    if solution[i] >= required_resource:
                        cnt += 1
                        solution[i] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            
        elif method == 'greedy-smooth':
            maximum_block = max(maximum_block_list)
            solution = [resource_num_list[i] for i in range(maximum_block)]
            resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
            resource_tuple_list.sort(key=lambda e: (-e[2], e[0] - e[1], -e[0]))
            for maximum_block, required_block, required_resource in resource_tuple_list:
                feasible, cnt = False, 0
                # idx_resource_pair = sorted([(idx, resource) for idx, resource in enumerate(solution)], key=lambda e: (e[1], e[0]), reverse=True)
                idx_resource_pair = sorted([(idx, resource) for idx, resource in enumerate(solution)], key=lambda e: (e[0], e[1]), reverse=True)

                for idx, resource in idx_resource_pair:
                    if idx < maximum_block and resource >= required_resource:
                        cnt += 1
                        solution[idx] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            
        return feasible, solution
    

    def fast_check_if_packable(self, required_resource, required_block, maximum_block, existing_solution, resource_num_list):
        if required_block > maximum_block: return False, existing_solution
        feasible_block = 0
        if len(existing_solution) < maximum_block:
            existing_solution += [resource_num_list[i] for i in range(len(existing_solution), maximum_block)]

        for i in range(maximum_block):
            if existing_solution[i] >= required_resource:
                feasible_block += 1
        
        feasible = feasible_block >= required_block
        if feasible:
            for i in range(maximum_block - 1, -1, -1):
                if existing_solution[i] >= required_resource and required_block > 0:
                    existing_solution[i] -= required_resource
                    required_block -= 1
                
        return feasible, existing_solution

    def batch_fast_check_if_packable(self, required_resource_dict, required_block_dict, maximum_block_dict, existing_solution_dict, resource_num_dict, user_list):
        solution = dict()
        for user in user_list:
            required_resource_list = required_resource_dict[user]
            required_block_list = required_block_dict[user]
            maximum_block_list = maximum_block_dict[user]
            existing_solution = existing_solution_dict[user]
            resource_num_list = resource_num_dict[user]

            solution[user] = list()
            if len(maximum_block_list) == 0:
                feasible = True
                solution[user] = copy.deepcopy(existing_solution)
                continue
            
            maximum_block = max(maximum_block_list)
            
            for i in range(maximum_block):
                if i < len(existing_solution): 
                    solution[user].append(existing_solution[i])
                else:
                    solution[user].append(resource_num_list[i])
            
            resource_tuple_list = [(a, b, c) for a, b, c in zip(maximum_block_list, required_block_list, required_resource_list)]
            # resource_tuple_list.sort(key=lambda e: (-e[2], e[0] - e[1], -e[0]))
            resource_tuple_list.sort(key=lambda e: (e[0], e[0] - e[1], e[2]))

            for maximum_block, required_block, required_resource in resource_tuple_list:
                feasible, cnt = False, 0
                for i in range(maximum_block - 1, -1, -1):
                    if solution[user][i] >= required_resource:
                        cnt += 1
                        solution[user][i] -= required_resource
                    if cnt == required_block:
                        feasible = True
                        break
                if not feasible: break
            if not feasible:
                solution = None
                break
        return feasible, solution
    
    
    def batch_fast_job_selection(self, required_resource_dict, required_block_dict, maximum_block_dict, existing_solution_dict, resource_num_dict, user_list):
        base_idx = 0
        solution_matrix = list()

        for user in user_list:
            required_resource_list = required_resource_dict[user]
            required_block_list = required_block_dict[user]
            maximum_block_list = maximum_block_dict[user]
            existing_solution = existing_solution_dict[user]
            resource_num_list = resource_num_dict[user]

            if len(maximum_block_list) == 0:
                feasible = True
                continue
            
            maximum_block = max(maximum_block_list)
            solution = list()
            for i in range(maximum_block):
                if i < len(existing_solution): 
                    solution.append(existing_solution[i])
                else:
                    solution.append(resource_num_list[i])

            idx_list = [i for i in range(len(maximum_block_list))]
            solution_matrix = solution_matrix + idx_list
            resource_tuple_list = [(a, b, c, i) for a, b, c, i in zip(maximum_block_list, required_block_list, required_resource_list, idx_list)]
            resource_tuple_list.sort(key=lambda e: ( -e[2], e[0] - e[1], -e[0]))
            
            for maximum_block, required_block, required_resource, idx in resource_tuple_list:
                feasible, cnt = False, 0
                cache_solution = [0 for _ in range(maximum_block)]
                for i in range(maximum_block - 1, -1, -1):
                    if solution[i] >= required_resource:
                        cnt += 1
                        solution[i] -= required_resource
                        cache_solution[i] = 1
                    if cnt == required_block:
                        feasible = True
                        break
                solution_matrix[idx + base_idx] = cache_solution
                if not feasible: break
            base_idx += len(resource_tuple_list)

            if not feasible:
                break
        return feasible, solution_matrix
    

    def job_selection(self, required_resource_dict, required_block_dict, maximum_block_dict, resource_num_dict, user_list, objective, max_seconds=5):
        required_block_list = list()
        maximum_block_list = list()
        required_resource_list = list()
        for user in user_list:
            if len(required_resource_dict[user]) > 0:
                required_block_list += required_block_dict[user]
                maximum_block_list += maximum_block_dict[user]
                required_resource_list += required_resource_dict[user]

        maximum_block = max(maximum_block_list)
        if isinstance(resource_num_list, int):
            resource_num_list = [resource_num_list for _ in range(maximum_block)]
        m = Model(solver_name=GRB)
        var_len = len(maximum_block_list) * maximum_block
        X = [m.add_var(var_type=BINARY) for i in range(len(maximum_block_list) * maximum_block)]
        if objective == 'random':
            m.objective = maximize(X[-1])
        elif objective == 'minimize':
            m.objective = maximize(-xsum(X[j] * required_resource_list[j // maximum_block]  for j in range(0, var_len, maximum_block)) )
        elif objective == 'maximize':
            m.objective = maximize(xsum(X[j] * required_resource_list[j // maximum_block]  for j in range(0, var_len, maximum_block)) )
        else:
            raise NotImplementedError

        # time-wise
        for i in range(len(required_resource_list)):
            m += xsum(X[j] for j in range(i * maximum_block, i * maximum_block + maximum_block_list[i])) == required_block_list[i]
            if maximum_block_list[i] <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + maximum_block_list[i], (i+1) * maximum_block)) == 0

        # resource-wise
        for i in range(maximum_block):
            base_j = 0
            for user in user_list:
                bottom = i + maximum_block * base_j
                top = bottom + maximum_block * (base_j + len(required_resource_dict[user]) + 1)
                m += xsum(X[j] * required_resource_list[j // maximum_block] for j in range(bottom, top, maximum_block) ) <= required_resource_dict[user][i]
                base_j += len(required_resource_dict[user])

        m.optimize(max_seconds=max_seconds)
        solution_matrix = list()
        for i in range(len(required_resource_list)):
            start = i * maximum_block
            solution = list()
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                solution.append(X[j].x)
            solution_matrix.append(solution)
        return solution_matrix




class SemiPreemptMIPSolver(object):
    def __init__(self, method):
        self.method = method

    def job_selection(self, required_resource_list, required_block_list, maximum_block_list, reward_list, existing_solution, resource_num_list, max_seconds=10):
        if len(maximum_block_list) == 0:
            return list()
        
        maximum_block = max(maximum_block_list)
        solution = list()
        for i in range(maximum_block):
            if i < len(existing_solution): 
                solution.append(existing_solution[i])
            else:
                solution.append(resource_num_list[i])
        if True:
            solution_matrix = list()
            for i in range(len(required_resource_list)):
                sol = None 
                # for j in range(maximum_block_list[i] - 1, required_block_list[i] - 1, -1):
                for j in range(required_block_list[i], maximum_block_list[i]+1):
                    left = j - required_block_list[i]
                    right = left + required_block_list[i]
                    done = True
                    for k in range(left, right):
                        # print(k, left, right, len(solution), maximum_block)
                        if solution[k] < required_resource_list[i]:
                            done = False
                    if done: 
                        sol = [0 for _ in range(maximum_block_list[i])]
                        for k in range(left, right):
                            sol[k] = 1
                            solution[k] -= required_resource_list[i]
                        break
                solution_matrix.append(sol)
            return solution_matrix
        
        
        m = Model(solver_name=GRB)
        X = [m.add_var(var_type=BINARY) for i in range(len(required_resource_list) * maximum_block)]
        obj_list = [X[i] * reward_list[i // maximum_block] for i in range(len(required_resource_list) * maximum_block)]
        m.objective = maximize(xsum(obj_list[i] for i in range(len(obj_list))))

        # job-wise
        for i in range(len(required_resource_list)):
            delta_len = maximum_block_list[i] - required_block_list[i] + 1
            m += xsum(X[j]  for j in range(i * maximum_block, i * maximum_block + delta_len)) <= 1
            
            if delta_len <  maximum_block:
                m += xsum(X[j] for j in range(i * maximum_block + delta_len, (i+1) * maximum_block)) == 0
        
        # resource-wise 
        for i in range(maximum_block):
            resource_list = list() 
            for j in range(len(required_block_list)):
                delta_len = maximum_block_list[j] - required_block_list[j] + 1
                for k in range(delta_len):
                    if k <= i and k + required_block_list[j] - 1 >= i:
                        resource_list.append(X[j * maximum_block + k] * required_resource_list[j]) # TODO
            m += xsum(resource_list) <= solution[i]

        m.optimize(max_seconds=max_seconds)
        solution_matrix = list()

        for i in range(len(required_resource_list)):
            start = i * maximum_block
            sol = list()
            start_idx = -1
            for j in range(start, start+maximum_block_list[i]):
                res = X[j].x
                if res is not None:
                    res = 0 if res < 0.5 else 1
                sol.append(res)
                if res == 1: start_idx = j

            assert sum(sol) <= 1, 'only allowed to select one solution'
            if sum(sol) == 0:
                solution_matrix.append(None)
            else:
                for j in range(start_idx, start_idx + required_block_list[i]):
                    sol[j - start] = 1
                solution_matrix.append(sol)
        return solution_matrix

