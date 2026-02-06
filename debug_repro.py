
import jax
import jax.numpy as jnp
from physax import make_config, create_ancestor_genome, init_organism, vm_step

def debug():
    cfg = make_config()
    genome, length = create_ancestor_genome(cfg)
    state = init_organism(genome, length, cfg)
    
    print(f"Initial State:")
    print(f"Genome Len: {state['genome_len']}")
    print(f"N Regs: {state['n_regs']}, N Instr: {state['n_instructions']}")
    print(f"IP: {state['ip']}, Code Start: {state['code_start']}")
    
    # Run for 200 steps
    for i in range(200):
        print(f"\nStep {i}: IP={state['ip']}, Regs={state['registers'][:4]}")
        state = vm_step(state, cfg)
        
        if state['has_child']:
            print(f"!!! BIRTH at step {i} !!!")
            print(f"Child Len: {state['child_len']}")
            print(f"Child Genome: {state['child_genome'][:50]}")
            break

if __name__ == "__main__":
    debug()
