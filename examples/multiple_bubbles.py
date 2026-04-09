"""Multiple-bubble test
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jbubble import SaveSpec, run_simulation
from jbubble.bubble.eom import ModifiedRayleighPlesset, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import NoShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jax import make_jaxpr
from jbubble.acoustics import IncompressibleMonopole




R0s = jnp.array([1.5e-6, 2.0e-6, 2.5e-6, 3.0e-6]) #(Nb,)


pulse = ToneBurst(freq=1e6, pressure=200e3, shape=Sine(), cycle_num=5) # --> needs to be a matrix (Nb,) after attenuation module is added
save_spec = SaveSpec(num_samples=2048)

# solve RP for number of bubbles
# eom = RayleighPlesset(
#     gas=PolytropicGas(gamma=1.4), 
#     shell=NoShell(sigma=0.072),
#     medium=NewtonianMedium(mu=0.001),
#     R0=R0s,
#     P_amb=101325.0,
#     rho_L=998.0,
#     # c_L=1500.0, 
# )

# result = run_simulation(eom, pulse, save_spec=save_spec)

# print(result.state.R.shape)  # (Nb, Nt)




def solve_one_bubble(R0: jax.Array):
    eom = ModifiedRayleighPlesset(
        gas=PolytropicGas(gamma=1.4),
        shell=NoShell(sigma=0.072),
        medium=NewtonianMedium(mu=0.001),
        R0=R0,
        P_amb=101325.0,
        rho_L=998.0,
        c_L=1500.0,
    )
    result_i = run_simulation(eom, pulse, save_spec=save_spec)
    return result_i.ts, result_i.state.R, result_i.state.R_dot, result_i.state_dot.R_dot


ts_all, R_all, R_dot_all, R_ddot_all = jax.vmap(solve_one_bubble)(R0s)

print(ts_all.shape)  # (Nb, Nt)
print(R_all.shape)  # (Nb, Nt)


#Plot results
t_us = ts_all * 1e6

fig, ax1 = plt.subplots(figsize=(10, 10))

for i, R0 in enumerate(R0s):
    label = f"bubble {i + 1}, R0={float(R0) * 1e6:.1f} um"
    ax1.plot(t_us[i], R_all[i] * 1e6, lw=1.8, label=label)
ax1.set_ylabel("Radius (um)")
ax1.set_title("Multiple Bubbles (Modified Rayleigh-Plesset)")
ax1.legend()
ax1.grid(True, alpha=0.3)



plt.tight_layout()
plt.show()
