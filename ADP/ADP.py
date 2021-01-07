import run_world
from run_world import read_grid, make_move, get_gamma, get_reward, get_next_states, is_goal, is_wall, not_goal_and_wall, \
    pretty_print_policy
import numpy as np
import copy

N_e = 30
R_plus = 2
WORLD1 = "lecture"
WORLD2 = "a4"



# explore(u,n): determine the value of the exploration function f
# given u and n
def explore(u, n):
    if n < N_e:
        return R_plus
    else:
        return u


# convertS(s): convert a state coordinates into list index
def convertS(s):
    a, b = s
    return a * 4 + b


# get_prob(n_sa, n_sas, curr_state, dir_intended, next_state):
# determine the transition probability based on counts
def get_prob(n_sa, n_sas, curr_state, dir_intended, next_state):
    a, b = curr_state
    d = n_sa[a][b][dir_intended]
    n = n_sas[convertS(curr_state)][dir_intended][next_state]
    if d == 0:
        return 0
    #print("prob:", n / d)
    return n / d

# remove_duplicate(l):
# remove the duplicate values of a list
def remove_duplicate(l):
    l.sort()
    return list(dict.fromkeys(l))

# possible_next(grid,curr_state):
# get all the possible next_state of the current state
def possible_next(grid, curr_state):
    ns = get_next_states(grid, curr_state)
    ns = remove_duplicate(ns)
    return ns


# exp_utils(grid, utils, curr_state, n_sa, n_sas):
# calculate the expected utilities Summation over s' of P(s'|s,a)U(s')
def exp_utils(grid, utils, curr_state, n_sa, n_sas):
    ns = possible_next(grid, curr_state)
    l, r, u, d = 0, 0, 0, 0
    for i in ns:
        u += get_prob(n_sa, n_sas, curr_state, 0, i) * utils[convertS(i)]
        r += get_prob(n_sa, n_sas, curr_state, 1, i) * utils[convertS(i)]        
        d += get_prob(n_sa, n_sas, curr_state, 2, i) * utils[convertS(i)] 
        l += get_prob(n_sa, n_sas, curr_state, 3, i) * utils[convertS(i)] 
    return u, r, d, l



# find_dir(val):
# find the optimal action from estimates utilities
def find_dir(val):
    dd = 0
    v = val[0]
    if v < val[1]:
        v = val[1]
        dd = 1
    if v < val[2]:
        v = val[2]
        dd = 2
    if v < val[3]:
        dd = 3
    return dd


# optimistic_exp_utils(grid, utils, curr_state, n_sa, n_sas):
# return the optimistic expected utilities
def optimistic_exp_utils(grid, utils, curr_state, n_sa, n_sas):
    u = exp_utils(grid, utils, curr_state, n_sa, n_sas)
    n = n_sa[curr_state[0]][curr_state[1]]

    result = []
    for i in range(0, 4):
        result.append(explore(u[i], n[i]))
    #print("optimistic_exp_utils")
    #print(result)
    return result


# update_utils(grid, utils, n_sa, n_sas, gamma,world):
# perform value iteration updates to the long-term
# expected utility estimates until the estimates converge
def update_utils(grid, utils, n_sa, n_sas, gamma, world):
    new_u = []
    c = 0
    for i in grid_coordinates:
        if not_goal_and_wall(grid,i):
            u = get_reward(world) + gamma * max(optimistic_exp_utils(grid, utils, i, n_sa, n_sas))
        else:
            u = utils[c]
        new_u.append(u)
        c += 1
    #print(new_u)
    return new_u


# utils_to_policy(grid, utils, n_sa, n_sas):
# determine the optimal policy given the current long-term utility value for each state
def utils_to_policy(grid, utils, n_sa, n_sas):
    p = []
    for i in grid_coordinates:
        p.append(find_dir(optimistic_exp_utils(grid, utils, i, n_sa, n_sas)))
    #print("utils to policy")
    #print(p)
    return p

# ADP(grid, utils, n_sa, n_sas, world):
# to run the ADP to determine the optimal policy and long term utility value
def ADP(grid, utils, n_sa, n_sas, world):
    s = (0,0)
    for z in range(0,50000):
        best_action = find_dir(optimistic_exp_utils(grid, utils, s, n_sa, n_sas))
        s_prime = make_move(grid, s, best_action, world)
        a, b = s
        n_sa[a][b][best_action] += 1
        n_sas[convertS(s)][best_action][s_prime] += 1
        utils = update_utils(grid, utils, n_sa, n_sas, get_gamma(world), world)
        s = s_prime
        if is_goal(grid,s):
            s = (0,0)
    return utils, utils_to_policy(grid,utils,n_sa,n_sas)


grid1 = read_grid(WORLD1)
grid2 = read_grid(WORLD2)

grid_coordinates = [(0, 0), (0, 1), (0, 2), (0, 3),
                    (1, 0), (1, 1), (1, 2), (1, 3),
                    (2, 0), (2, 1), (2, 2), (2, 3)]

n_sa1 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

n_sa2 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

u_1 = [0, 0, 0, 0,
       0, 0, 0, -1,
       0, 0, 0, 1]

u_2 = [0, 0, 0, 0,
       0, 0, 0, 0,
       0, -1, 1, 0]


#initialize n_sas1
n_sas1 = [[], [], [], [], [], [], [], [], [], [], [], []]

count = 0
for i in grid_coordinates:
    ns = get_next_states(grid1, i)
    ns = remove_duplicate(ns)
    dic = {}
    for j in ns:
        dic.update({j: 0})

    for i in range(0, 4):
        dic2 = copy.deepcopy(dic)
        n_sas1[count].append(dic2)
    count += 1

#initialize n_sas2
n_sas2 = [[], [], [], [], [], [], [], [], [], [], [], []]

count = 0
for i in grid_coordinates:
    ns = get_next_states(grid2, i)
    ns = remove_duplicate(ns)
    dic = {}
    for j in ns:
        dic.update({j: 0})

    for i in range(0, 4):
        dic2 = copy.deepcopy(dic)
        n_sas2[count].append(dic2)
    count += 1



u,p = ADP(grid1,u_1,n_sa1,n_sas1,WORLD1)
for i in range(0,12):
    u[i] = round(float(u[i]),6)
arr = np.array(u)
arr = arr.reshape(3,4)
print("The long term utility values for grid lecture:")
print(arr)
arr = np.array(p)
arr = arr.reshape(3,4)
print()
print("The optimal policy for grid lecture:")
pretty_print_policy(grid1, arr)


u,p = ADP(grid2,u_2,n_sa2,n_sas2,WORLD2)
for i in range(0,12):
    u[i] = round(float(u[i]),6)
arr = np.array(u)
arr = arr.reshape(3,4)
print()
print("The long term utility values for grid a4:")
print(arr)
arr = np.array(p)
arr = arr.reshape(3,4)
print()
print("The optimal policy for grid a4:")
pretty_print_policy(grid2, arr)

