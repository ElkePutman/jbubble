"""01 Basic Simulation

The absolute minimal microbubble simulation.

We simulate a 2 micron radius air bubble in water, driven by a
1 MHz acoustic pulse at 50 kPa. This uses the high-level
'run_simulation' API which handles the integration and
returns a 'SimulationResult' object.
"""

import jax
import matplotlib.pyplot as plt
from jbubble import run_simulation
from jbubble.bubble.eom import ModifiedRayleighPlesset, RayleighPlesset
from jbubble.bubble.gas import PolytropicGas
from jbubble.bubble.medium import NewtonianMedium
from jbubble.bubble.shell import MarmottantSurfaceTension, NoShell, LipidShell
from jbubble.pulse import ToneBurst
from jbubble.pulse.shapes import Sine
from jbubble.solver import SaveSpec

from jax import make_jaxpr

# 1. Define physics components
# jbubble uses a 'composition' approach: you build an Equation of Motion (EoM)
# by picking a gas law, a shell model, and a medium model.
gas = PolytropicGas(gamma=1.07)
# shell = NoShell(sigma=0.072)  # Surface tension of water (0.072 N/m)
shell = LipidShell(sigma=MarmottantSurfaceTension(R_buckle_ratio=0.98, chi=0.5, sigma_rupture=0.072), kappa_s=1.0e-9)
medium = NewtonianMedium(mu=0.001)  # Viscosity of water (0.001 Pa s)

# 2. Build the Equation of Motion (EoM)
eom = ModifiedRayleighPlesset(
    gas=gas,
    shell=shell,
    medium=medium,
    R0=2.0e-6,  # 2 micron equilibrium radius
    P_amb=101325.0,  # 1 atm ambient pressure
    rho_L=998.0,  # Density of water (kg/m^3)
    c_L =1480.0,  # Speed of sound in water (m/s)
)

# 3. Define the acoustic driving pulse
# Here we use a 1 MHz ToneBurst with a Sine shape, lasting for 5 cycles.
pulse = ToneBurst(
    freq=1e6,
    pressure=50e3,
    shape=Sine(),
    cycle_num=5,
)

# 4. Run the simulation
# We JIT compile 'run_simulation' for maximum performance.
# 'SaveSpec' controls how many points are saved in the result.
result = jax.jit(run_simulation)(
    eom,
    pulse,
    save_spec=SaveSpec(num_samples=1000),
)



# 5. Visualize the radius over time
fig,(ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax2.plot(result.ts * 1e6, result.radius * 1e6, lw=2, color="navy")
ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Radius (µm)")
ax2.set_title("2 µm Air Bubble in Water (1 MHz, 50 kPa)")
ax2.grid(True, alpha=0.3)



ax1.plot(result.ts * 1e6, pulse(result.ts), lw=2, color="crimson")
ax1.set_xlabel("Time (µs)")
ax1.set_ylabel("Amplitude")
ax1.set_title("Acoustic Driving Pulse")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


