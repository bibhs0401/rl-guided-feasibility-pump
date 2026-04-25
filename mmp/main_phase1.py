import random
import numpy as np
import time
from docplex.mp.model import Model
random.seed(10)


# extract data from csv file and store in the respective variables
def required_data(data):
    # number of constraints or rows of matrix A
    m = int(data.loc[0]['m'])
    # print(" m = ", m)

    # number of decision variables or number of columns in matrix A
    n = int(data.loc[0]['n'])

    # number of multiplicative objective functions
    p = int(data.loc[0]['p'])
    d = list(data.loc[0:p - 1]['d'])

    # cost vector (row vector)
    c = []
    for i in range(p):
        c.append(list(data.loc[i * n:((i + 1) * n - 1)]['c']))
        # for i in range(m):
        # c.append(0)
    # c = np.array(c)
    # print("Cost Vector is: ",c)

    # b vector or the RHS vector
    b = np.array(data.loc[0:m - 1]['b'])
    # print("Initial Solution vector is: ",b)

    # decision variables coefficient matrix
    # A = np.zeros((m,n))
    A = np.array(data.loc[:]['A']).reshape(m, n)

    # lOOP FOR ASSIGNING ELEMENTS TO MATRIX A
    #     k = 0
    #     for i in range(m):
    #         for j in range(n):
    #             A[i][j] = int(data.loc[k]['A'])
    #             k += 1

    # print("Matrix A in standard form in standard form is:\n", A)

    return A, b, c, m, n, d, p


def get_integer_index(num_of_dec_var):
    # generally in andrea lodi's paper 90% of variables are integer in average
    # I = sorted(list(set(random.sample(range(num_of_dec_var), int(9 * num_of_dec_var/ 10)))))
    I = [i for i in range(int(0.8 * num_of_dec_var))]
    not_I = []
    for j in range(num_of_dec_var):
        if j not in I:
            not_I.append(j)

    return I, not_I


def FP_parameters(I, m, n):
    # time interval in seconds CPU time = 1800 CPU seconds (parameter from Feasibility pump paper)
    time_limit = np.log(m + n) / 4

    # average number of variables to be flipped T (set from feasibility pump paper) 30 was initial value
    average_number_of_variable_to_be_flipped = 10

    # perturberation frequency parameter (R) i.e perterburation after every 100 iterations....... nIT
    perturberation_frequency = 100

    # exact number of integer variables to be flipped TT
    exact_num_of_int_var_to_flip = random.randint(int(average_number_of_variable_to_be_flipped / 2),
                                                  int(3 * average_number_of_variable_to_be_flipped / 2))

    if exact_num_of_int_var_to_flip > len(I):
        exact_num_of_int_var_to_flip = average_number_of_variable_to_be_flipped

    return time_limit, average_number_of_variable_to_be_flipped, exact_num_of_int_var_to_flip, perturberation_frequency

def MILMMP_parameters():
    epsilon = 100
    multiplier = 10**4
    return epsilon, multiplier


def first_linear_model(A, b, c, d, n, p, I):
    m1 = Model()
    m1.context.cplex_parameters.threads = 1
    # m1.Params.Threads = 1

    x = m1.continuous_var_matrix(1, n, lb=0, name="x")
    multiplicative_y = []
    for i in range(p):
        multiplicative_y.append(m1.continuous_var(name='y{}'.format(i)))
    m1.add_constraints(sum(x[0, j] * A[k][j] for j in range(n)) <= b[k] for k in range(len(A)))
    m1.add_constraints(x[0, j] >= 0 for j in range(n))
    m1.add_constraints(x[0, j] <= 1 for j in I)

    for k in range(p):
        lhs = 0
        for j in range(n):
            lhs += c[k][j] * x[0, j]
        m1.add_constraint(lhs + d[k] == multiplicative_y[k])
    m1.set_objective('max', sum(multiplicative_y[k] for k in range(p)))
    return m1, multiplicative_y

def check_infeasibility(model):
    if model.solve_details.status_code == 3:
        return True   # true means the model is infeasible
    else:
        return False


def solve_first_linear_model(m1, n):
    # m1.setParam('OutputFlag', 0)
    m1.solve()
    infeasible = check_infeasibility(m1)
    if infeasible == True:
        return ('infeasible', 'none', 'none', 'none')
    else:
        x_y_list = []
        for i in m1.iter_variables():
            x_y_list.append(i.solution_value)
            # x_relaxed = x*

        z_lp = m1.objective_value  # objective values

        x_relaxed, y_values = [], []  # y values is the list of values of Y for p number of objectives
        for v in range(n):
            x_relaxed.append(float(x_y_list[v]))
        for u in range(n, len(x_y_list)):
            y_values.append(x_y_list[u])

        return m1, z_lp, x_relaxed, y_values

# check whether the floating point number (in our case: values of decision variables) is integer?
# I = index position where the list x_relaxed must have integer values

def check_integer(I, x_relaxed):   ### function improved
    true_count = 0
    for items in I:
        if x_relaxed[items].is_integer():
            true_count += 1
        elif not x_relaxed[items].is_integer():
            return False
    if len(I) == true_count:
        return True
    else:
        return False

def rounding(x_relaxed, I):  ### function improved
    x_tilde = []
    for count in range(len(x_relaxed)):
        if count in I:
            x_tilde.append(round(x_relaxed[count]))   # x_tilde dimension is similar to x*
        else:
            x_tilde.append(x_relaxed[count])
    return x_tilde


# this function only creates the distance based first model
def second_model_FP(n, A, b, c, d, p, I):
    f = Model()
    f.context.cplex_parameters.threads = 1
    z = f.continuous_var_matrix(1, n, lb=0, name="z")
    multiplicative_y = []
    for i in range(p):
        multiplicative_y.append(f.continuous_var(name='y{}'.format(i)))
    objvar = f.continuous_var()

    for k in range(p):
        lhs = 0

        for j in range(n):
            lhs += c[k][j] * z[0, j]
        f.add_constraint(lhs + d[k] == multiplicative_y[k])

    f.add_constraints(sum(z[0, j] * A[i][j] for j in range(n)) <= b[i] for i in range(len(A)))
    f.add_constraints(z[0, i] >= 0 for i in I)
    f.add_constraints(z[0, i] <= 1 for i in I)
    # f.addConstrs(sum(z[0,j] for j in I if x_tilde[j] == 0) + sum(1-z[0,j] for j in I if x_tilde[j]==1) == objvar)
    f.set_objective('min', objvar)
    return f, objvar, z, multiplicative_y


# this function solves distance based second model
def solve_second_model_FP(f, z, n):
    # f.setParam('OutputFlag', 0)
    # f.setObjective(sum(z[0,j] for j in I if x_tilde[j] == 0) + sum(1-z[0,j] for j in I if x_tilde[j]==1), GRB.MINIMIZE)
    f.solve()
    # delta = f.getObjective()
    # delta = delta.getValue()

    infeasible = check_infeasibility(f)
    if infeasible == True:
        return ('infeasible', 'none', 'none', 'none')
    else:
        x_y_list = []
        for i in f.iter_variables():
            x_y_list.append(i.solution_value)
        # x_list is the list containing values of x decision variables and y decision variables
        # y values is the list of values of Y for p number of objectives
        x_list, y_values = [], []
        for v in range(n):
            x_list.append(float(x_y_list[v]))
        for u in range(n, len(x_y_list) - 1):
            y_values.append(x_y_list[u])
        delta = f.objective_value
        return f, delta, x_list, y_values

def feasibility_pump(f, I,c,d,TT,n, objvar, z, time_limit, x_tilde, y_values):
    nIT = 0
    FP_start_time = time.time()
    while ((time.time() - FP_start_time) < time_limit):
        nIT += 1
        # print("nIT = ", nIT)

        # calling second_distance_based_model_FP
        objfun = f.add_constraint(objvar >= sum(z[0, j] for j in I if x_tilde[j] == 0) + sum(1 - z[0, j] for j in I if x_tilde[j] == 1))
        f, delta, x_list, y_values = solve_second_model_FP(f, z, n)
        if y_values == 'none':
            break
        f.remove_constraint(objfun)
        boolean = check_integer(I, x_list)
        if boolean == True:
            z_ip = sum((np.dot(c[i], x_list) + d[i]) for i in range(len(c)))
            return (f, x_list, y_values, z_ip, delta, time.time() - FP_start_time, nIT)

        if nIT <= 100:
            count = 0
            for items in I:
                if round(x_list[items]) != x_tilde[items]:
                    count += 1
                    if count >= 1:
                        break
            if count >= 1:
                for i in I:
                    x_tilde = rounding(x_list, I)
            else:
                distance_list = []
                for items in I:
                    distance_list.append(abs(x_list[items] - x_tilde[items]))
                if len(distance_list) > 0 and TT <= len(I):
                    temp_list = []
                    for i in range(TT):
                        max_index = np.argmax(distance_list)  # maximum distance index in distance_list
                        temp_list.append(I[max_index])
                        distance_list[max_index] = - 100
                    for items in temp_list:
                        if x_tilde[items] == 0:
                            x_tilde[items] = 1
                        elif x_tilde[items] == 1:
                            x_tilde[items] = 0
        elif nIT > 100:  ##################### need to change the code for checking last 3 iterations as well, not only after 100 iterations
            for j in I:
                ro = np.random.uniform(-0.3, 0.7)
                if (abs(x_list[j] - x_tilde[j]) + max(ro, 0)) > 0.5:
                    if x_tilde[j] == 0:
                        x_tilde[j] = 1
                    elif x_tilde[j] == 1:
                        x_tilde[j] = 0

    return (f, 'none', 'none', 'none', delta, time.time() - FP_start_time, nIT)

def check_for_perturbation(prev_three_x):
    if (np.array_equiv(prev_three_x[0], prev_three_x[1]) == True) and (np.array_equiv(prev_three_x[1], prev_three_x[2]) == True):
        return True
    else:
        return False

def cut_parameters(y_values, p, epsilon, multiplier):
    cut_list = []
    cut_rhs = []

    for counter in range(len(y_values)):
        cut_list.append(multiplier * (1 / y_values[counter]))

        # cut_matrix = np.array(cut_matrix)
    cut_rhs.append(epsilon + (multiplier * p))  ## epsilon + (multiplier*p)
    # cut_rhs = np.array(cut_rhs)
    return cut_list, cut_rhs

def add_cut(model, cut_list, multiplicative_y, cut_rhs, cut_iteration, phase_iteration):
    model.add_constraint(sum(cut_list[i] * multiplicative_y[i] for i in range(len(multiplicative_y))) >= cut_rhs[0], ctname = 'cut{}_{}'.format(cut_iteration, phase_iteration))
    #model.addMConstr(cut_matrix, multiplicative_y, '>', cut_rhs, name = 'cut_constraint')
    return model

def check_infeasibility(model):
    if model.solve_details.status_code == 3:
    #model.getAttr(GRB.Attr.Status) == 3:
        print("INFEASIBLE !!!!!!!")
        return True   # true means the model is infeasible
    elif model.solve_details.status_code == 5:
        print("UNBOUNDED!!!!!!!!!!")
        return True
    else:
        return False

def check_for_termination(multiplicative_y):
    for items in multiplicative_y:
        if items == 0 or items <= 0:
            return True  #true will stop the multiplicative programming
        else:
            continue
    return False         # false continues the multiplicative programming


def multiplicative_FP(start_time, z, n ,I, c, d, TT, TL,previous_y_values, m1, f, objvar):

    nIT_list = []  # number of FP iteration in each cut iteration
    cut_iteration = 0

    for i in range(1):

        # LP relaxation with help of gurobi
        m1, z_lp, x_relaxed, y_values = solve_first_linear_model(m1, n)
        if y_values == 'none':
            delta = 'none'
            decision_variables, algorithm_objective_value, solution_time, nIT = x_relaxed, z_lp, time.time() - start_time, 0
            #nIT_list.append(nIT)

            return (m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration)

        is_it_integer = check_integer(I, x_relaxed)
        if is_it_integer == True:
            delta = 0
            decision_variables, algorithm_objective_value, solution_time, nIT = x_relaxed, z_lp, time.time() - start_time, 0
            #nIT_list.append(nIT)

        else:
            x_tilde = rounding(x_relaxed, I)
            f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT = feasibility_pump(f,I,c,d,TT,n,
                                                                                                                     objvar,
                                                                                                                     z,
                                                                                                                     TL,
                                                                                                                     x_tilde,
                                                                                                                     y_values)
            #nIT_list.append(nIT)
            if decision_variables == 'none':
                return m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration
            #print("cut_iteration = ", cut_iteration + 1)
            print(y_values)
        previous_y_values = y_values
        """
        termination = check_for_termination(y_values)
        if termination == True:
            break
        cut_iteration += 1
        cut_list, cut_rhs = cut_parameters(y_values, p, epsilon, multiplier)
        m1 = add_cut(m1, cut_list, multiplicative_y_m1, cut_rhs, cut_iteration, phase_iteration)
        f = add_cut(f, cut_list, multiplicative_y_f, cut_rhs, cut_iteration, phase_iteration)
        # print("x ",decision_variables)        
        """


    return m1, f, decision_variables, previous_y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration
#***********************




def main_function(A, b, c, m, n, d, p):

#with open("output.csv","w", newline = '') as csv:
    #csv.write("decision_variables, y_values, optimal_value, delta, solution_time, FP_iterations, cut_iterations")

    #read and get data

# parameters
    epsilon, multiplier = MILMMP_parameters()
    I, not_I = get_integer_index(n)
    TL, T, TT, R = FP_parameters(I, m, n)

    y_values = [0 for i in range(p)]    # previous y_values (for starting the algorithm)
    print("value of n", n)
    m1, multiplicative_y_m1 = first_linear_model(A, b, c, d, n, p, I)
    f, objvar, z, multiplicative_y_f = second_model_FP(n, A, b, c, d, p, I)

    start_time = time.time()
    #for p3 in range(1):
    #print("***phase", (p3 + 1))
    # try:
    m1, f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = multiplicative_FP(start_time, z, n, I, c, d, TT, TL, y_values, m1, f, objvar)
    """    
        for cut_itr in range(0, cut_iteration):
            m1.get_constraint_by_name("cut{}_{}".format(cut_itr + 1, p3 + 1)).rhs = m1.get_constraint_by_name(
                "cut{}_{}".format(cut_itr + 1, p3 + 1)).rhs - epsilon
    # except:
    # print(type(m1))
    # else:
        if p == 2:
            m1.objective_expr.set_coefficients([(m1.get_var_by_name("y0"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y1"), max(0.1, np.random.standard_normal() + 1))])
        if p == 3:
            m1.objective_expr.set_coefficients([(m1.get_var_by_name("y0"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y1"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y2"), max(0.1, np.random.standard_normal() + 1))])
        if p == 4:
            m1.objective_expr.set_coefficients([(m1.get_var_by_name("y0"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y1"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y2"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y3"), max(0.1, np.random.standard_normal() + 1))])
        if p == 5:
            m1.objective_expr.set_coefficients([(m1.get_var_by_name("y0"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y1"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y2"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y3"), max(0.1, np.random.standard_normal() + 1)),
                                                (m1.get_var_by_name("y4"), max(0.1, np.random.standard_normal() + 1))])
    """
    solution_time = time.time() - start_time

    #final_list = [decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration]

#with open("output.csv", "a", newline='') as csv:
    #csv.write("\n")
    #csv.write(str(final_list))

#print(solution_time)
    return decision_variables, y_values, algorithm_objective_value, delta, solution_time,nIT_list, cut_iteration
