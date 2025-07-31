import cvxpy as cp
import numpy as np
from collections import OrderedDict

def global_min_var_cvxpy(mu, S, w_prev, b_h, k, lamb, c_h, allow_short, tickers=None):
    mu = np.asarray(mu)
    S = np.asarray(S)
    c_h = np.asarray(c_h)
    N = len(mu)

    # Define variable
    w = cp.Variable(N)

    # Objective: variance + transaction cost
    variance = cp.quad_form(w, cp.psd_wrap(S))
    if w_prev is not None:
        t_cost = lamb * cp.sum(cp.multiply(c_h, cp.abs(w - w_prev)))
    else:
        t_cost = 0

    objective = cp.Minimize(variance + t_cost)

    # Constraints
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= b_h,
        cp.norm1(w) <= k
    ]
    if allow_short:
        constraints += [w >= -1, w <= 1]
    else:
        constraints += [w >= 0, w <= 1]

    # Problem definition and solving
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)  

    # Return result
    if w.value is not None:
        weights = w.value
        if tickers is not None:
            return OrderedDict(zip(tickers, weights))
        return weights
    else:
        print("CVXPY Optimization failed:", problem.status)
        if tickers is not None:
            return OrderedDict(zip(tickers, w_prev if w_prev is not None else np.ones(N) / N))
        return w_prev if w_prev is not None else np.ones(N) / N


#____________________________________________________________________________________________________________________________

def max_sharpe_modified_cvxpy(mu, S, k, allow_short, w_prev, lamb, c_h, tickers=None):
    mu = np.asarray(mu)
    S = np.asarray(S)
    c_h = np.asarray(c_h)
    N = len(mu)    

    y = cp.Variable(N)
    alpha = cp.Variable(pos=True)

    variance = cp.quad_form(y, cp.psd_wrap(S))

    if w_prev is not None:
        w_prev=np.asarray(w_prev)
        t_cost = lamb * cp.sum(cp.multiply(c_h, cp.abs(y - alpha * w_prev)))
        
    else:
        t_cost = 0

    # Constraints
    constraints = [ mu @ y == 1,                    
        cp.norm1(y) <= alpha*k,
        cp.sum(y) == alpha]
    
    if not allow_short:
        constraints.append(y >= 0)
    else:
        constraints += [y >= -1, y <= 1]

    
    objective = cp.Minimize(variance+t_cost)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    if y.value is not None:
        # Normalize y to get weights summing to 1
        weights = y.value / alpha.value
        if tickers is not None:
            return OrderedDict(zip(tickers, weights))
        return weights
    else:
        print("Optimization failed:", prob.status)
        if tickers is not None:
            return OrderedDict(zip(tickers, w_prev if w_prev is not None else np.ones(N) / N))
        return w_prev if w_prev is not None else np.ones(N) / N
