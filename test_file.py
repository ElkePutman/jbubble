import jbubble
from jbubble.utils.presets import free_bubble
import jax

preset = free_bubble()
from jbubble import run_simulation, SaveSpec
result = jax.jit(run_simulation)(
    preset.eom, preset.pulse,
    save_spec=SaveSpec(num_samples=500),
    t_max=10e-6,
)
print("converged:", bool(result.converged))
print("peak R/R0:", float(result.radius.max() / preset.eom.R0))