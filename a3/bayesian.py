import numpy as np

'''AB: There is food left in Aria’s food bowl.
 AH: Aria howls.
AS: Aria is sick.
 M: There is a full moon.
 NA: Your neighbour is away.
 NH: Your neighbour’s dog howls '''

P_AS = 0.05
f_AS = [['AS'], [1, 0.05], [0, 0.95]]

P_M = 1 / 28
f_M = [['M'], [1, 1 / 28], [0, 27 / 28]]

P_NA = 0.4
f_NA = [['NA'], [1, 0.4], [0, 0.6]]

P_AB_cond_notAS = 0.2
P_AB_cond_AS = 0.8
f_AB_AS = [['AB', 'AS'], [1, 1, 0.8], [1, 0, 0.2], [0, 0, 0.8], [0, 1, 0.2]]

P_AH_cond_AS_M_NH = 0.95
P_AH_cond_AS_M_notNH = 0.85
P_AH_cond_AS_notM_NH = 0.70
P_AH_cond_AS_notM_notNH = 0.55
P_AH_cond_notAS_M_NH = 0.65
P_AH_cond_notAS_M_notNH = 0.30
P_AH_cond_notAS_notM_NH = 0.15
P_AH_cond_notAS_notM_notNH = 0
f_AH_AS_M_NH = [['AH', 'AS', 'M', 'NH'], [1, 1, 1, 1, 0.95], [1, 1, 1, 0, 0.85], [1, 1, 0, 1, 0.7],
                [1, 1, 0, 0, 0.55], [1, 0, 1, 1, 0.65], [1, 0, 1, 0, 0.3], [1, 0, 0, 1, 0.15],
                [1, 0, 0, 0, 0.0], [0, 1, 1, 1, 0.05], [0, 1, 1, 0, 0.15], [0, 1, 0, 1, 0.3],
                [0, 1, 0, 0, 0.45], [0, 0, 1, 1, 0.35], [0, 0, 1, 0, 0.7], [0, 0, 0, 1, 0.85],
                [0, 0, 0, 0, 1]]

P_NH_cond_M_NA = 0.9
P_NH_cond_M_notNA = 0.3
P_NH_cond_notM_NA = 0.6
P_NH_cond_notM_notNA = 0
f_M_NA_NH = [['M', 'NA', 'NH'], [1, 1, 1, 0.9], [1, 1, 0, 0.1], [1, 0, 1, 0.3], [1, 0, 0, 0.7], [0, 1, 1, 0.6],
             [0, 1, 0, 0.4], [0, 0, 1, 0], [0, 0, 0, 1]]


# restrict(factor, variable, value): restricts the given variable in the given factor to
# the given value
def restrict(factor, variable, value):
    str1 = " "

    # return string
    str1 = (str1.join(factor[0]))

    s = "Restrict f(" + str1 + ") to " + variable + " = " + str(value) + " to produce f("

    index = 0
    new_factor = []
    v = []
    prob = []
    for a in factor[0]:
        if a == variable:
            i = 1
            while i < len(factor):
                f = factor[i]
                if f[index] == value:
                    f.pop(index)
                    prob.append(f)
                i += 1
        else:
            # break
            v.append(a)
            index += 1
    str1 = " "

    s += str1.join(v) + ")"
    new_factor.append(v)
    for i in prob:
        new_factor.append(i)
    print(s)
    # for i in new_factor:
    print_factor(new_factor)
    return new_factor


# multiply(factora, factorb): performs pointwise multiplication of the given two factors
def multiply(factora, factorb):
    len1 = len(factora[0])
    len2 = len(factorb[0])

    v = sorted(factora[0] + factorb[0])

    common_v = []
    i, j = 0, 1
    length = len(v)
    while j < length:
        if v[i] == v[j]:
            common_v.append(v[i])
        i += 1
        j += 1

    # print("common v: " + str(common_v))

    indexa = []
    indexb = []
    for i in common_v:
        indexa.append(factora[0].index(i))
        indexb.append(factorb[0].index(i))

    # print("variable indexa:" + str(indexa))
    # print("variable indexb:" + str(indexb))
    # new variable name
    v = list(set().union(factora[0], factorb[0]))
    v.sort()
    new_factor = []
    new_factor.append(v)

    factor_order = []

    # print("v: " + str(v))
    for i in v:
        if i in factora[0]:
            factor_order.append([0, factora[0].index(i)])
        else:
            factor_order.append([1, factorb[0].index(i)])
    # print("factor_order: " + str(factor_order))

    # for each row in factora, compare to each row in factorb
    # if the shared variable are the same, do element wise multiplication
    # multiply the probability, keep the rest of the variable the same

    i, j = 1, 1
    num1 = len(factora)
    num2 = len(factorb)

    while i < num1:
        f1 = factora[i]
        a = []
        for val in indexa:
            a.append(f1[val])

        p = f1[len1]

        while j < num2:
            prob = []
            f2 = factorb[j]

            b = []
            for val in indexb:
                b.append(f2[val])
            if a == b:
                for ff in factor_order:
                    if ff[0] == 0:
                        prob.append(f1[ff[1]])
                    else:
                        prob.append(f2[ff[1]])
                p2 = p * f2[len2]
                prob.append(p2)
                new_factor.append(prob)

            j += 1
        i += 1
        j = 1

    return new_factor

def factorname(f):
    length = len(f)
    var_name = f[0]
    if length == 2:
        string = "f() "
    else:
        string = "f("
        for i in var_name:
            string += i + " "

        string = string[:len(string)-1]
        string += ") "
    return string

def tablename(f):
    length = len(f)
    var_name = f[0]
    string = ""
    if length == 2:
        string = "Prob"
    #  print(f[1][0])
    else:
        for i in var_name:
            string += i + ","
        string += " Prob"
    return string


def print_factor(f):
    length = len(f)
    print(tablename(f))
    if length == 2:
        print(f[1][0])
    else:
        j = 1
        while j < length:
            factor = f[j]
            string = ""

            a = 0
            l = len(factor)
            while a <l-1:
                val = factor[a]
                a+=1
                if val == 1:
                    string += "True,"
                elif val == 0:
                    string += "False,"
            string += str(factor[l-1])
            j += 1
            print(string)


# sumout(factor, variable): sums out a variable in a given factor
def sumout(factor, variable):
    index = 0
    for i in factor[0]:
        if i == variable:
            break
        index += 1
    len_f = len(factor[0])
    str1 = ' '
    str1 = (str1.join(factor[0]))

    string = "Sum out " + str(variable) + " from " + factorname(factor) + " to produce "

    new_v = []
    for i in factor[0]:
        new_v.append(i)
    new_v.pop(index)

    new_factor = []
    new_factor.append(new_v)

    str1 = ' '
    str1 += (str1.join(new_v))
    print(string + factorname(new_factor))
    length = len(factor) - (len(factor) - 1) / 2

    while len(factor) > 1:
        f = factor[1]
        val = f[index]
        for i in factor:
            prob = []

            if isinstance(i[index], int) and val + i[index] == 1:
                a = 0
                while a < len_f:
                    if a != index:
                        if i[a] != f[a]:
                            break
                    a += 1
                if a == len_f:
                    for b in i:
                        prob.append(b)
                    prob[len_f] = f[len_f] + i[len_f]
                    prob.pop(index)
                    new_factor.append(prob)
                    factor.remove(f)
                    factor.remove(i)
                    break

    # print("sum out factor: ", str(new_factor))
    print_factor(new_factor)
    return new_factor


# normalize(factor): normalizes a factor by dividing each entry by the
# sum of all the entries
def normalize(factor):
    length = len(factor[0])
    sum = 0

    sum += factor[1][length]
    sum += factor[2][length]

    factor[1][length] = factor[1][length] / sum
    factor[2][length] = factor[2][length] / sum

    print_factor(factor)
    return factor


# ve(factor_list, query_variables, ordered_list_hidden_variables, evidence_list):
# computes Pr(query_variables|evidence_vars) by the VEA
def ve(factor_list, query_variables, ordered_list_hidden_variables, evidence_list):
    length = len(factor_list)
    # restrict evident_variable
    to_be_removed = []
    j = 0
    while j < length:
        flag = False
        f = factor_list[j]
        for var in evidence_list:
            if var[0] in f[0]:
                to_be_removed.append(j)
                f = restrict(f, var[0], var[1])
                print ("")
                flag = True
        if (flag):
            factor_list.append(f)
        # if i contains variable in the evidence_list
        # restrict
        j += 1

    temp = []
    l = len(factor_list)
    for i in range(l):
        if i not in to_be_removed:
            temp.append(factor_list[i])
    factor_list = temp

    length = len(factor_list)

    # multiply/sum out hidden variables
    for i in ordered_list_hidden_variables:
        if i != query_variables:
            flag = False
            factors_containing_v = []
            to_be_removed = []
            j = 0
            while j < length:
                f = factor_list[j]
                if i in f[0]:
                    factors_containing_v.append(f)
                    to_be_removed.append(j)
                    flag = True
                j += 1

            temp = []
            l = len(factor_list)
            for nn in range(l):
                if nn not in to_be_removed:
                    temp.append(factor_list[nn])
            factor_list = temp


            if len(factors_containing_v) > 1:
                v = []
                string = "Multiply "
                for i2 in factors_containing_v:
                    v = list(set().union(i2[0],v))
                    string += factorname(i2)
                v.sort()
                string += "to produce f("
                for i2 in v:
                    string += i2 + " "
                string = string[:len(string)-1]
                string+= ")"
                print(string)

            multi_flag = False
            while len(factors_containing_v) > 1:
                f1 = factors_containing_v.pop(0)
                f2 = factors_containing_v.pop(0)
                factors_containing_v.append(multiply(f1, f2))
                multi_flag = True
            if multi_flag:
                print_factor(factors_containing_v[0])
                print("")
            if flag:
                factor_list.append(sumout(factors_containing_v[0], i))
                print("")
            length = len(factor_list)
            # factors_containing_v.pop(0)


    # multiply remaining factor
    if len(factor_list) > 1:
        v = []
        string = "Multiply "
        for i2 in factor_list:
            v = list(set().union(i2[0], v))
            string += factorname(i2)
        v.sort()
        string += "to produce f("
        for i2 in v:
            string += i2 + " "
        string = string[:len(string) - 1]
        string += ")"
        print(string)

    flag = False
    while len(factor_list) > 1:
        f1 = factor_list.pop(0)
        f2 = factor_list.pop(0)
        factor_list.append(multiply(f1, f2))
        flag = True
    if flag:
        print_factor(factor_list[0])
        print("")
    # no more hidden variables in the list
    # multiply all the remaining factors



    # normalize
    # normalize the last factor
    name = factorname(factor_list[0])
    string = "Normalize " + name + " to produce " + name
    print(string)
    f = normalize(factor_list[0])

    print("")
    # return the query_variable
    return f


# part A
print("Part A:")
print("Computing P(AS | AB and AH)")
print ("Define factors f(AB AS) f(AH AS M NH) f(AS) f(M) f(M NA NH) f(NA)")
print("")
factor_list = [f_AB_AS, f_AH_AS_M_NH, f_AS, f_M, f_M_NA_NH, f_NA]
query_variables = 'AS'
ordered_list_hidden_variables = ['AB', 'AH', 'AS', 'M', 'NA', 'NH']
evidence_list = [['AB', 1], ['AH', 1]]
result = ve(factor_list, query_variables, ordered_list_hidden_variables, evidence_list)


# part B
print("Part B:")
print("Computing P(AS | AB and AH and M and not NA)")
print ("Define factors f(AB AS) f(AH AS M NH) f(AS) f(M) f(M NA NH) f(NA)")
print("")
f_AS = [['AS'], [1, 0.05], [0, 0.95]]
f_M = [['M'], [1, 1 / 28], [0, 27 / 28]]
f_NA = [['NA'], [1, 0.4], [0, 0.6]]
f_AB_AS = [['AB', 'AS'], [1, 1, 0.8], [1, 0, 0.2], [0, 0, 0.8], [0, 1, 0.2]]
f_AH_AS_M_NH = [['AH', 'AS', 'M', 'NH'], [1, 1, 1, 1, 0.95], [1, 1, 1, 0, 0.85], [1, 1, 0, 1, 0.7],
                [1, 1, 0, 0, 0.55], [1, 0, 1, 1, 0.65], [1, 0, 1, 0, 0.3], [1, 0, 0, 1, 0.15],
                [1, 0, 0, 0, 0.0], [0, 1, 1, 1, 0.05], [0, 1, 1, 0, 0.15], [0, 1, 0, 1, 0.3],
                [0, 1, 0, 0, 0.45], [0, 0, 1, 1, 0.35], [0, 0, 1, 0, 0.7], [0, 0, 0, 1, 0.85],
                [0, 0, 0, 0, 1]]
f_M_NA_NH = [['M', 'NA', 'NH'], [1, 1, 1, 0.9], [1, 1, 0, 0.1], [1, 0, 1, 0.3], [1, 0, 0, 0.7], [0, 1, 1, 0.6],
             [0, 1, 0, 0.4], [0, 0, 1, 0], [0, 0, 0, 1]]
factor_list = [f_AB_AS, f_AH_AS_M_NH, f_AS, f_M, f_M_NA_NH, f_NA]
query_variables = 'AS'
ordered_list_hidden_variables = ['AB', 'AH', 'AS', 'M', 'NA', 'NH']
evidence_list = [['AB', 1], ['AH', 1], ['M', 1], ['NA', 0]]
result = ve(factor_list, query_variables, ordered_list_hidden_variables, evidence_list)

