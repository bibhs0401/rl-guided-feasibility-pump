import logging
from collections import defaultdict
import numpy as np
import time
import random
from docplex.mp.model import Model

random.seed(10)
np.random.seed(10)

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger("phase1_rl")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# =========================================================
# DATA / PARAMETERS
# =========================================================
def required_data(data):
    # number of constraints or rows of matrix A
    m = int(data.loc[0]['m'])

    # number of decision variables
    n = int(data.loc[0]['n'])

    # number of multiplicative objective functions
    p = int(data.loc[0]['p'])
    d = list(data.loc[0:p - 1]['d'])

    # cost vector
    c = []
    for i in range(p):
        c.append(list(data.loc[i * n:((i + 1) * n - 1)]['c']))

    # RHS vector
    b = np.array(data.loc[0:m - 1]['b'])

    # coefficient matrix
    A = np.array(data.loc[:]['A']).reshape(m, n)

    return A, b, c, m, n, d, p


def get_integer_index(num_of_dec_var):
    # same as your baseline: first 80% treated as integer
    I = [i for i in range(int(0.8 * num_of_dec_var))]
    not_I = []
    for j in range(num_of_dec_var):
        if j not in I:
            not_I.append(j)
    return I, not_I


def FP_parameters(I, m, n):
    # time limit
    time_limit = np.log(m + n) / 4

    # average number of variables to be flipped
    average_number_of_variable_to_be_flipped = 10

    # perturbation frequency parameter
    perturberation_frequency = 100

    # exact number of integer variables to be flipped
    exact_num_of_int_var_to_flip = random.randint(
        int(average_number_of_variable_to_be_flipped / 2),
        int(3 * average_number_of_variable_to_be_flipped / 2)
    )

    if exact_num_of_int_var_to_flip > len(I):
        exact_num_of_int_var_to_flip = average_number_of_variable_to_be_flipped

    return (
        time_limit,
        average_number_of_variable_to_be_flipped,
        exact_num_of_int_var_to_flip,
        perturberation_frequency
    )


def MILMMP_parameters():
    epsilon = 100
    multiplier = 10 ** 4
    return epsilon, multiplier


# =========================================================
# RL AGENT + HELPERS
# =========================================================
class Phase1FlipAgent:
    """
    Q-learning agent.
    Action = choose ONE variable to flip from top candidates when stalling occurs.
    """
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(float)

    def choose_action(self, state, candidates):
        if not candidates:
            return None

        if np.random.rand() < self.epsilon:
            return random.choice(candidates)

        qvals = [self.q[(state, a)] for a in candidates]
        return candidates[int(np.argmax(qvals))]

    def update(self, state, action, reward, next_state, next_candidates):
        if action is None:
            return

        if not next_candidates:
            best_next = 0.0
        else:
            best_next = max(self.q[(next_state, a)] for a in next_candidates)

        old_q = self.q[(state, action)]
        self.q[(state, action)] = old_q + self.alpha * (
            reward + self.gamma * best_next - old_q
        )


def build_state(x_list, x_tilde, I, nIT):
    """
    Compact discretized state for tabular Q-learning.
    """
    if len(I) == 0:
        return (0, 0, 0, 0)

    frac = [abs(x_list[j] - round(x_list[j])) for j in I]
    mean_frac = np.mean(frac) if len(frac) > 0 else 0.0
    max_frac = np.max(frac) if len(frac) > 0 else 0.0
    mismatch_ratio = np.mean([1 if round(x_list[j]) != x_tilde[j] else 0 for j in I])

    mean_bin = min(4, int(mean_frac * 10))
    max_bin = min(4, int(max_frac * 10))
    mismatch_bin = min(4, int(mismatch_ratio * 10))
    iter_bin = min(4, int(nIT / 20))

    return (mean_bin, max_bin, mismatch_bin, iter_bin)


def candidate_variables(x_list, x_tilde, I, top_k=10):
    """
    Candidate variables ranked by |x_j - x_tilde_j|.
    """
    pairs = [(j, abs(x_list[j] - x_tilde[j])) for j in I]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [j for j, _ in pairs[:min(top_k, len(pairs))]]


def flip_one(x_tilde, j):
    if j is None:
        return x_tilde

    if x_tilde[j] == 0:
        x_tilde[j] = 1
    elif x_tilde[j] == 1:
        x_tilde[j] = 0
    return x_tilde


# =========================================================
# MODELS
# =========================================================
def first_linear_model(A, b, c, d, n, p, I):
    m1 = Model()
    m1.context.cplex_parameters.threads = 1

    x = m1.continuous_var_matrix(1, n, lb=0, name="x")
    multiplicative_y = []

    for i in range(p):
        multiplicative_y.append(m1.continuous_var(name=f'y{i}'))

    m1.add_constraints(
        sum(x[0, j] * A[k][j] for j in range(n)) <= b[k]
        for k in range(len(A))
    )
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
        logger.warning("Model infeasible")
        return True
    elif model.solve_details.status_code == 5:
        logger.warning("Model unbounded")
        return True
    else:
        return False


def solve_first_linear_model(m1, n):
    m1.solve()
    infeasible = check_infeasibility(m1)

    if infeasible is True:
        return ('infeasible', 'none', 'none', 'none')

    x_y_list = []
    for i in m1.iter_variables():
        x_y_list.append(i.solution_value)

    z_lp = m1.objective_value

    x_relaxed, y_values = [], []
    for v in range(n):
        x_relaxed.append(float(x_y_list[v]))
    for u in range(n, len(x_y_list)):
        y_values.append(x_y_list[u])

    return m1, z_lp, x_relaxed, y_values


def check_integer(I, x_relaxed):
    true_count = 0
    for items in I:
        if x_relaxed[items].is_integer():
            true_count += 1
        else:
            return False
    return len(I) == true_count


def rounding(x_relaxed, I):
    x_tilde = []
    for count in range(len(x_relaxed)):
        if count in I:
            x_tilde.append(round(x_relaxed[count]))
        else:
            x_tilde.append(x_relaxed[count])
    return x_tilde


def second_model_FP(n, A, b, c, d, p, I):
    f = Model()
    f.context.cplex_parameters.threads = 1

    z = f.continuous_var_matrix(1, n, lb=0, name="z")
    multiplicative_y = []

    for i in range(p):
        multiplicative_y.append(f.continuous_var(name=f'y{i}'))

    objvar = f.continuous_var()

    for k in range(p):
        lhs = 0
        for j in range(n):
            lhs += c[k][j] * z[0, j]
        f.add_constraint(lhs + d[k] == multiplicative_y[k])

    f.add_constraints(
        sum(z[0, j] * A[i][j] for j in range(n)) <= b[i]
        for i in range(len(A))
    )
    f.add_constraints(z[0, i] >= 0 for i in I)
    f.add_constraints(z[0, i] <= 1 for i in I)

    f.set_objective('min', objvar)
    return f, objvar, z, multiplicative_y


def solve_second_model_FP(f, z, n):
    f.solve()
    infeasible = check_infeasibility(f)

    if infeasible is True:
        return ('infeasible', 'none', 'none', 'none')

    x_y_list = []
    for i in f.iter_variables():
        x_y_list.append(i.solution_value)

    x_list, y_values = [], []
    for v in range(n):
        x_list.append(float(x_y_list[v]))
    for u in range(n, len(x_y_list) - 1):
        y_values.append(x_y_list[u])

    delta = f.objective_value
    return f, delta, x_list, y_values


# =========================================================
# FEASIBILITY PUMP WITH RL STALL HANDLING
# =========================================================
def feasibility_pump(
    f, I, c, d, TT, n, objvar, z, time_limit, x_tilde, y_values,
    phase1_agent=None
):
    """
    Baseline FP logic is kept.
    Only the 'stalling flip' block is changed:
    - instead of flipping TT largest-distance variables,
      RL chooses ONE variable to flip.
    """
    if phase1_agent is None:
        phase1_agent = Phase1FlipAgent()

    nIT = 0
    FP_start_time = time.time()
    delta = None

    # pending transition for delayed reward:
    # (state_before_flip, action_var, delta_before_flip)
    pending_transition = None

    logger.info(f"FP started | time_limit={time_limit:.2f}s")

    while (time.time() - FP_start_time) < time_limit:
        nIT += 1

        objfun = f.add_constraint(
            objvar >= sum(z[0, j] for j in I if x_tilde[j] == 0)
                   + sum(1 - z[0, j] for j in I if x_tilde[j] == 1)
        )

        f, delta, x_list, y_values = solve_second_model_FP(f, z, n)

        if y_values == 'none':
            logger.warning(f"FP stopped: no y_values at nIT={nIT}")
            break

        f.remove_constraint(objfun)

        # If there was a previous RL flip, now we can score it.
        # When the current point is integer-feasible, add a terminal bonus
        # before clearing the pending transition.
        state_now = build_state(x_list, x_tilde, I, nIT)
        candidates_now = candidate_variables(x_list, x_tilde, I, top_k=10)
        boolean = check_integer(I, x_list)

        if pending_transition is not None:
            prev_state, prev_action, prev_delta = pending_transition
            reward = prev_delta - delta
            next_candidates = candidates_now
            if boolean is True:
                reward += 10.0
                next_candidates = []

            phase1_agent.update(prev_state, prev_action, reward, state_now, next_candidates)
            logger.info(
                f"RL update | nIT={nIT} | action={prev_action} | "
                f"prev_delta={prev_delta:.6f} | new_delta={delta:.6f} | reward={reward:.6f}"
            )
            pending_transition = None

        if boolean is True:
            z_ip = sum((np.dot(c[i], x_list) + d[i]) for i in range(len(c)))

            logger.info(
                f"Integer solution found | nIT={nIT} | delta={delta:.6f} | "
                f"elapsed={time.time() - FP_start_time:.2f}s"
            )
            return (f, x_list, y_values, z_ip, delta, time.time() - FP_start_time, nIT)

        if nIT <= 100:
            count = 0
            for items in I:
                if round(x_list[items]) != x_tilde[items]:
                    count += 1
                    if count >= 1:
                        break

            # -----------------------
            # normal FP progress
            # -----------------------
            if count >= 1:
                x_tilde = rounding(x_list, I)
                logger.info(
                    f"nIT={nIT} | rounding updated | delta={delta:.6f} | "
                    f"elapsed={time.time() - FP_start_time:.2f}s"
                )

            # -----------------------
            # stalling: RL decides which variable to flip
            # -----------------------
            else:
                state = build_state(x_list, x_tilde, I, nIT)
                candidates = candidate_variables(x_list, x_tilde, I, top_k=10)

                chosen_var = phase1_agent.choose_action(state, candidates)
                x_tilde = flip_one(x_tilde, chosen_var)

                pending_transition = (state, chosen_var, delta)

                logger.info(
                    f"STALL detected | nIT={nIT} | delta={delta:.6f} | "
                    f"chosen_var={chosen_var} | candidates={candidates}"
                )

        else:
            # keep your original random perturbation rule after many iterations
            logger.info(f"Large-iteration perturbation | nIT={nIT}")
            for j in I:
                ro = np.random.uniform(-0.3, 0.7)
                if (abs(x_list[j] - x_tilde[j]) + max(ro, 0)) > 0.5:
                    if x_tilde[j] == 0:
                        x_tilde[j] = 1
                    elif x_tilde[j] == 1:
                        x_tilde[j] = 0

    logger.warning(
        f"FP timed out | nIT={nIT} | elapsed={time.time() - FP_start_time:.2f}s"
    )
    return (f, 'none', 'none', 'none', delta, time.time() - FP_start_time, nIT)


# =========================================================
# CUT HELPERS
# =========================================================
def check_for_perturbation(prev_three_x):
    if (
        np.array_equiv(prev_three_x[0], prev_three_x[1]) is True
        and np.array_equiv(prev_three_x[1], prev_three_x[2]) is True
    ):
        return True
    else:
        return False


def cut_parameters(y_values, p, epsilon, multiplier):
    cut_list = []
    cut_rhs = []

    for counter in range(len(y_values)):
        cut_list.append(multiplier * (1 / y_values[counter]))

    cut_rhs.append(epsilon + (multiplier * p))
    return cut_list, cut_rhs


def add_cut(model, cut_list, multiplicative_y, cut_rhs, cut_iteration, phase_iteration):
    model.add_constraint(
        sum(cut_list[i] * multiplicative_y[i] for i in range(len(multiplicative_y))) >= cut_rhs[0],
        ctname=f'cut{cut_iteration}_{phase_iteration}'
    )
    return model


def check_for_termination(multiplicative_y):
    for items in multiplicative_y:
        if items == 0 or items <= 0:
            return True
    return False


# =========================================================
# OUTER FP WRAPPER
# =========================================================
def multiplicative_FP(
    start_time, z, n, I, c, d, TT, TL, previous_y_values, m1, f, objvar,
    phase1_agent=None
):
    nIT_list = []
    cut_iteration = 0

    for _ in range(1):
        m1, z_lp, x_relaxed, y_values = solve_first_linear_model(m1, n)

        if y_values == 'none':
            delta = 'none'
            decision_variables = x_relaxed
            algorithm_objective_value = z_lp
            solution_time = time.time() - start_time
            nIT = 0

            return (
                m1, f, decision_variables, previous_y_values,
                algorithm_objective_value, delta, solution_time,
                nIT_list, cut_iteration
            )

        is_it_integer = check_integer(I, x_relaxed)

        if is_it_integer is True:
            delta = 0
            decision_variables = x_relaxed
            algorithm_objective_value = z_lp
            solution_time = time.time() - start_time
            nIT = 0

        else:
            x_tilde = rounding(x_relaxed, I)

            f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT = feasibility_pump(
                f, I, c, d, TT, n, objvar, z, TL, x_tilde, y_values,
                phase1_agent=phase1_agent
            )

            nIT_list.append(nIT)

            if decision_variables == 'none':
                return (
                    m1, f, decision_variables, previous_y_values,
                    algorithm_objective_value, delta, solution_time,
                    nIT_list, cut_iteration
                )

            logger.info(f"FP returned y_values={y_values}")

        previous_y_values = y_values

    return (
        m1, f, decision_variables, previous_y_values,
        algorithm_objective_value, delta, solution_time,
        nIT_list, cut_iteration
    )


# =========================================================
# MAIN ENTRY
# =========================================================
def main_function(A, b, c, m, n, d, p, phase1_agent=None):
    epsilon, multiplier = MILMMP_parameters()
    I, not_I = get_integer_index(n)
    TL, T, TT, R = FP_parameters(I, m, n)

    y_values = [0 for _ in range(p)]

    logger.info("=" * 80)
    logger.info(f"main_function started | m={m} | n={n} | p={p} | TL={TL:.2f}")

    m1, multiplicative_y_m1 = first_linear_model(A, b, c, d, n, p, I)
    f, objvar, z, multiplicative_y_f = second_model_FP(n, A, b, c, d, p, I)

    start_time = time.time()

    m1, f, decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration = multiplicative_FP(
        start_time, z, n, I, c, d, TT, TL, y_values, m1, f, objvar,
        phase1_agent=phase1_agent
    )

    solution_time = time.time() - start_time

    logger.info(
        f"main_function finished | time={solution_time:.2f}s | "
        f"obj={algorithm_objective_value} | cut_iteration={cut_iteration}"
    )

    return decision_variables, y_values, algorithm_objective_value, delta, solution_time, nIT_list, cut_iteration
