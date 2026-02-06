
import jax
import jax.numpy as jnp
import numpy as np
from physax import make_config, init_population, cycle_step

def run_verify():
    cfg = make_config()
    cfg.pop_size = 16  # Small grid (4x4)
    cfg.initial_pop = 1
    cfg.max_age = 1000
    
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    
    # Init logic directly
    pop = init_population(k1, cfg)
     
    print("Initial Population:")
    print(f"Alive: {jnp.sum(pop['alive'])}")
    alive_idx = jnp.argmax(pop['alive'])
    print(f"Stats for organism {alive_idx}:")
    print(f"  IP: {pop['ip'][alive_idx]}")
    print(f"  Age: {pop['age'][alive_idx]}")
    print(f"  Genome Len: {pop['genome_len'][alive_idx]}")
    
    # Jit the step
    step_fn = jax.jit(cycle_step, static_argnums=(0,))
    
    keys = jax.random.split(k2, 600)
    
    for i in range(600):
        pop, stats = step_fn(cfg, pop, keys[i])
        
        # Check specific organism
        if pop['alive'][alive_idx]:
             ip = pop['ip'][alive_idx]
             age = pop['age'][alive_idx]
             regs = pop['registers'][alive_idx]
             color = pop['color'][alive_idx]
             has_child = pop['has_child'][alive_idx]
             child_len = pop['child_len'][alive_idx]
             if i % 10 == 0 or has_child:
                 print(f"Step {i}: IP={ip}, Regs={regs[:4]}, Color={color}, HasChild={has_child}, ChildLen={child_len}")
                 
        births = stats['births']
        if births > 0:
            print(f"!!! BIRTH DETECTED at Step {i} !!! Total Births: {births}")
            print(f"Alive Count: {jnp.sum(pop['alive'])}")
            # Pick a new alive one to track maybe?
            
if __name__ == "__main__":
    run_verify()
