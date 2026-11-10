import numpy as np


def check_schedule(gen, g, schedule, tol=1e-6, verbose=True):
    """Check one generator's schedule against every rule, without Gurobi.

    Returns True if all rules hold; otherwise prints each failure and returns False.
    """
    p = np.asarray(schedule["p"], dtype=float)
    u = np.asarray(schedule["u"])
    v = np.asarray(schedule["v"])
    w = np.asarray(schedule["w"])
    T = len(p)

    min_out     = float(gen["min_out"][g])
    max_out     = float(gen["max_out"][g])
    min_up      = int(gen["min_up"][g])
    min_down    = int(gen["min_down"][g])
    op_rampup   = float(gen["op_rampup"][g])
    op_rampdown = float(gen["op_rampdown"][g])
    st_rampup   = float(gen["st_rampup"][g])
    sh_rampdown = float(gen["sh_rampdown"][g])

    errors = []

    # 1. binaries
    for name, x in (("u", u), ("v", v), ("w", w)):
        for t in range(T):
            if x[t] not in (0, 1):
                errors.append(f"[binary]    hour {t}: {name} = {x[t]} is not 0 or 1")

    # 2. output limits
    for t in range(T):
        if u[t] == 0 and abs(p[t]) > tol:
            errors.append(f"[output]    hour {t}: off but p = {p[t]:.4f}")
        if u[t] == 1 and not (min_out - tol <= p[t] <= max_out + tol):
            errors.append(f"[output]    hour {t}: on but p = {p[t]:.4f} "
                          f"not in [{min_out}, {max_out}]")

    # 3. start/stop logic (on before the horizon)
    for t in range(T):
        u_prev = 1 if t == 0 else u[t - 1]
        if u[t] - u_prev != v[t] - w[t]:
            errors.append(f"[logic]     hour {t}: u {u_prev}->{u[t]} but v = {v[t]}, w = {w[t]}")

    # 4. no double switch
    for t in range(T):
        if v[t] + w[t] > 1:
            errors.append(f"[switch]    hour {t}: v and w both 1")

    # 5. min up, checked forwards: after a start, stay on min_up hours
    for t in range(T):
        if v[t] == 1:
            for s in range(t, min(t + min_up, T)):
                if u[s] != 1:
                    errors.append(f"[min_up]    start at hour {t}, but off at hour {s} "
                                  f"(min_up = {min_up})")
                    break

    # 6. min down, checked forwards: after a shut-down, stay off min_down hours
    for t in range(T):
        if w[t] == 1:
            for s in range(t, min(t + min_down, T)):
                if u[s] != 0:
                    errors.append(f"[min_down]  shut-down at hour {t}, but on at hour {s} "
                                  f"(min_down = {min_down})")
                    break

    # 7-8. ramp limits (t >= 1)
    for t in range(1, T):
        rise = p[t] - p[t - 1]
        up_lim = op_rampup * u[t - 1] + st_rampup * v[t]
        if rise > up_lim + tol:
            errors.append(f"[ramp_up]   hour {t}: rise {rise:.4f} > limit {up_lim:.4f}")
        fall = p[t - 1] - p[t]
        down_lim = op_rampdown * u[t] + sh_rampdown * w[t]
        if fall > down_lim + tol:
            errors.append(f"[ramp_down] hour {t}: fall {fall:.4f} > limit {down_lim:.4f}")

    if verbose:
        for e in errors:
            print(f"  gen {g} FAIL {e}")
    return len(errors) == 0


def schedule_value(gen, g, schedule, y_d, y_r):
    """Priced objective of a schedule, computed directly (should equal solve's value)."""
    p = np.asarray(schedule["p"], dtype=float)
    u = np.asarray(schedule["u"], dtype=float)
    v = np.asarray(schedule["v"], dtype=float)
    y_d = np.asarray(y_d, dtype=float)
    y_r = np.asarray(y_r, dtype=float)

    max_out  = float(gen["max_out"][g])
    nol_cost = float(gen["nol_cost"][g])
    mar_cost = float(gen["mar_cost"][g])
    st_cost  = float(gen["st_cost"][g])

    cost = nol_cost * u + mar_cost * p + st_cost * v
    payment = y_d * p + y_r * (max_out * u - p)
    return float(np.sum(cost - payment))
