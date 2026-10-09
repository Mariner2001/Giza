"""Numerical checks of the quantities quoted in Sect. 4 of the paper (analysis of the GPC update)."""
import math
import json
import numpy as np

g, theta, mu_min, mu_max = 9.8, math.radians(10.0), 1.0, 10.0
beta = 1.0 / (2 * g * math.sin(theta))
alpha_max = 1.0 / (2 * g * (math.sin(theta) + mu_min * math.cos(theta)))
alpha_min = 1.0 / (2 * g * (math.sin(theta) + mu_max * math.cos(theta)))
out = dict(beta=beta, alpha_min=alpha_min, alpha_max=alpha_max)

# Proposition 2(c): lower bound for rho = 1/2 on the Sphere box (R = 5.12)
R, rho = 5.12, 0.5
out["bound_rho_half"] = math.sqrt(rho / (2 * beta * R))
out["bound_valid_for_abs_p_at_least"] = rho / (2 * beta * R) * alpha_max
c = rho / (beta * R)
out["exact_prob_without_displacement"] = 2 * math.sqrt(c) - c

# Monte Carlo confirmation of the exact probability (alpha = 0) and of the clipping fraction for Griewank
rng = np.random.default_rng(0)
M = 2_000_000
v0 = rng.random(M)
mu = mu_min + (mu_max - mu_min) * rng.random(M)
d = v0 ** 2 / (2 * g * (math.sin(theta) + mu * math.cos(theta)))
x = v0 ** 2 / (2 * g * math.sin(theta))
eps = rng.uniform(-R, R, M)
p = 1.0
y = (p + d) * (x * eps)
out["mc_prob_contract_p1"] = float(np.mean(np.abs(y) <= rho * p))
eps600 = rng.uniform(-600, 600, M)
y600 = (100.0 + d) * (x * eps600)
out["griewank_clipped_fraction_p100"] = float(np.mean(np.abs(y600) > 600))

# Corollary 1 example: |t_j| = 2.56 (a quarter of the Sphere box width), N = 1000
out["corollary_example_bound"] = 2.56 ** 2 / (64 * 1000 ** 2)

print(json.dumps(out, indent=1))
json.dump(out, open("../results/theory_checks.json", "w"), indent=1)
