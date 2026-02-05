"""
Physis Minimal - JAX Version

Parallelized digital evolution simulation using JAX.
- Population runs in parallel via vmap
- Generations via lax.scan
- Fixed-size padded genomes for JAX compatibility
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random, vmap
from functools import partial
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle

# Enable 64-bit precision if needed
# jax.config.update("jax_enable_x64", True)


# ==========================================
# 1. CONSTANTS (Gene values)
# ==========================================

# Structural genes
R = 0
S = 1
B = 2
I = 3
SEP = 4

# Atomic operations
MOVE = 10
LOAD = 11
STORE = 12
JUMP = 20
IFZERO = 21
INC = 30
DEC = 31
ADD = 32
SUB = 33
ALLOCATE = 40
DIVIDE = 41
READ_SIZE = 50

# Special values
NOP = -1  # No operation (padding)
EMPTY = -1  # Empty slot marker


# ==========================================
# 2. CONFIGURATION
# ==========================================

class Config:
    """Simulation configuration with fixed sizes for JAX."""
    max_genome_len: int = 100      # Maximum genome length
    max_registers: int = 8         # Maximum registers per organism
    max_instructions: int = 20     # Maximum instruction definitions
    max_ops_per_instr: int = 10    # Maximum ops per instruction
    max_code_len: int = 50         # Maximum code section length
    pop_size: int = 1000           # Population size
    cpu_cycles: int = 500          # CPU cycles per organism per epoch
    max_age: int = 80              # Maximum age before death
    
    # Mutation rates
    point_mutation_rate: float = 0.01
    indel_rate: float = 0.005
    

def make_config(**kwargs) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ==========================================
# 3. ORGANISM STATE (as JAX arrays)
# ==========================================

def create_organism_state(cfg: Config):
    """Create empty organism state arrays."""
    return {
        'genome': jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
        'genome_len': jnp.int32(0),
        'registers': jnp.zeros(cfg.max_registers, dtype=jnp.int32),
        'ip': jnp.int32(0),
        'code_start': jnp.int32(0),
        'n_regs': jnp.int32(1),
        'n_instructions': jnp.int32(0),
        'age': jnp.int32(0),
        'alive': jnp.bool_(True),
        'has_child': jnp.bool_(False),
        'child_genome': jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
        'child_len': jnp.int32(0),
        'child_write_ptr': jnp.int32(0),
        # Parsed instruction table: [instr_idx, op_idx] -> (op_code, arg0, arg1)
        'instr_ops': jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32),
        'instr_args': jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32),
        'instr_n_ops': jnp.zeros(cfg.max_instructions, dtype=jnp.int32),
    }


# ==========================================
# 4. GENOME PARSING (compile-time phenotype)
# ==========================================

def parse_genome(genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """
    Parse genome into phenotype (registers, instructions, code_start).
    Returns parsed instruction table for execution.
    """
    # Initialize outputs
    n_regs = jnp.int32(0)
    ptr = jnp.int32(0)
    
    # A. Parse hardware section (count R's until B, I, or SEP)
    def count_regs(carry, x):
        ptr, n_regs, done = carry
        is_R = (genome[ptr] == R)
        is_S = (genome[ptr] == S)
        is_B = (genome[ptr] == B)
        is_I = (genome[ptr] == I)
        is_SEP = (genome[ptr] == SEP)
        is_end = is_B | is_I | is_SEP | (ptr >= genome_len)
        
        n_regs = jnp.where(is_R & ~done, n_regs + 1, n_regs)
        ptr = jnp.where(~done, ptr + 1, ptr)
        ptr = jnp.where(is_B & ~done, ptr, ptr)  # B advances ptr
        done = done | is_end
        
        return (ptr, n_regs, done), None
    
    (ptr, n_regs, _), _ = lax.scan(count_regs, (jnp.int32(0), jnp.int32(0), False), jnp.arange(cfg.max_genome_len))
    
    # Ensure at least 1 register
    n_regs = jnp.maximum(n_regs, 1)
    
    # Skip past B marker if present
    ptr = jnp.where(ptr > 0, ptr, 0)
    language_start = ptr
    
    # B. Parse instruction definitions
    instr_ops = jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32)
    instr_args = jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32)
    instr_n_ops = jnp.zeros(cfg.max_instructions, dtype=jnp.int32)
    
    def parse_instructions(carry, _):
        ptr, instr_idx, instr_ops, instr_args, instr_n_ops, done = carry
        
        # Check if we hit SEP or end
        at_sep = (ptr < genome_len) & (genome[ptr] == SEP)
        at_end = (ptr >= genome_len) | done
        
        # Check if at instruction marker
        at_I = (ptr < genome_len) & (genome[ptr] == I) & ~at_sep & ~at_end
        
        # Parse one instruction if at I marker
        def parse_one_instr(state):
            ptr, op_idx, ops_row, args_row = state
            ptr = ptr + 1  # Skip I marker
            
            def parse_ops(carry, _):
                ptr, op_idx, ops_row, args_row, done = carry
                
                at_end = (ptr >= genome_len) | done
                gene = jnp.where(at_end, NOP, genome[ptr])
                is_marker = (gene == I) | (gene == SEP)
                
                # Determine arity
                is_2arg = (gene == MOVE) | (gene == LOAD) | (gene == STORE) | (gene == ADD) | (gene == SUB)
                is_1arg = (gene == READ_SIZE) | (gene == ALLOCATE) | (gene == INC) | (gene == DEC) | (gene == JUMP) | (gene == IFZERO)
                is_0arg = (gene == DIVIDE)
                arity = jnp.where(is_2arg, 2, jnp.where(is_1arg, 1, jnp.where(is_0arg, 0, 1)))
                
                # Store op and args
                valid_op = ~is_marker & ~at_end & (op_idx < cfg.max_ops_per_instr)
                ops_row = jnp.where(valid_op, ops_row.at[op_idx].set(gene), ops_row)
                
                # Read args
                arg0_ptr = ptr + 1
                arg1_ptr = ptr + 2
                arg0 = jnp.where((arg0_ptr < genome_len) & valid_op, genome[arg0_ptr], 0)
                arg1 = jnp.where((arg1_ptr < genome_len) & valid_op & is_2arg, genome[arg1_ptr], 0)
                
                args_row = jnp.where(valid_op, args_row.at[op_idx, 0].set(arg0), args_row)
                args_row = jnp.where(valid_op, args_row.at[op_idx, 1].set(arg1), args_row)
                
                # Advance pointer
                advance = jnp.where(is_2arg, 3, jnp.where(is_1arg, 2, 1))
                ptr = jnp.where(valid_op, ptr + advance, ptr)
                op_idx = jnp.where(valid_op, op_idx + 1, op_idx)
                done = done | is_marker | at_end
                
                return (ptr, op_idx, ops_row, args_row, done), None
            
            (ptr, op_idx, ops_row, args_row, _), _ = lax.scan(
                parse_ops, 
                (ptr, jnp.int32(0), ops_row, args_row, False), 
                jnp.arange(cfg.max_ops_per_instr)
            )
            return ptr, op_idx, ops_row, args_row
        
        # Default no-op
        ops_row = instr_ops[instr_idx]
        args_row = instr_args[instr_idx]
        
        new_ptr, n_ops, ops_row, args_row = lax.cond(
            at_I,
            parse_one_instr,
            lambda s: (s[0] + 1, jnp.int32(0), s[2], s[3]),  # Skip non-I
            (ptr, jnp.int32(0), ops_row, args_row)
        )
        
        # Update arrays
        instr_ops = jnp.where(at_I, instr_ops.at[instr_idx].set(ops_row), instr_ops)
        instr_args = jnp.where(at_I, instr_args.at[instr_idx].set(args_row), instr_args)
        instr_n_ops = jnp.where(at_I, instr_n_ops.at[instr_idx].set(n_ops), instr_n_ops)
        
        ptr = jnp.where(at_I, new_ptr, jnp.where(~at_sep & ~at_end, ptr + 1, ptr))
        instr_idx = jnp.where(at_I, instr_idx + 1, instr_idx)
        done = done | at_sep | (instr_idx >= cfg.max_instructions)
        
        return (ptr, instr_idx, instr_ops, instr_args, instr_n_ops, done), None
    
    (ptr, n_instructions, instr_ops, instr_args, instr_n_ops, _), _ = lax.scan(
        parse_instructions,
        (language_start, jnp.int32(0), instr_ops, instr_args, instr_n_ops, False),
        jnp.arange(cfg.max_genome_len)
    )
    
    # Skip past SEP
    code_start = jnp.where((ptr < genome_len) & (genome[ptr] == SEP), ptr + 1, ptr)
    
    return {
        'n_regs': n_regs,
        'n_instructions': jnp.maximum(n_instructions, 1),
        'code_start': code_start,
        'instr_ops': instr_ops,
        'instr_args': instr_args,
        'instr_n_ops': instr_n_ops,
    }


# ==========================================
# 5. VM EXECUTION (single step)
# ==========================================

def vm_step(state: dict, cfg: Config):
    """Execute one VM step for an organism."""
    genome = state['genome']
    genome_len = state['genome_len']
    registers = state['registers']
    ip = state['ip']
    code_start = state['code_start']
    n_regs = state['n_regs']
    n_instructions = state['n_instructions']
    instr_ops = state['instr_ops']
    instr_args = state['instr_args']
    child_genome = state['child_genome']
    child_len = state['child_len']
    child_write_ptr = state['child_write_ptr']
    has_child = state['has_child']
    alive = state['alive']
    
    # Compute code length
    code_len = genome_len - code_start
    valid_code = code_len > 0
    
    # Wrap IP if needed
    ip = jnp.where(
        valid_code & ((ip < code_start) | (ip >= genome_len)),
        code_start,
        ip
    )
    
    # Fetch opcode
    op_code = jnp.where(valid_code & (ip < genome_len), genome[ip], 0)
    ip = ip + 1
    ip = jnp.where(ip >= genome_len, code_start, ip)
    
    # Get instruction index
    instr_idx = jnp.where(n_instructions > 0, op_code % n_instructions, 0)
    
    # Execute all ops in this instruction
    def exec_op(carry, op_idx):
        registers, child_genome, child_len, child_write_ptr, has_child, ip = carry
        
        op = instr_ops[instr_idx, op_idx]
        arg0 = instr_args[instr_idx, op_idx, 0]
        arg1 = instr_args[instr_idx, op_idx, 1]
        
        valid_op = op != NOP
        
        # Helper to get register value
        def get_reg(idx):
            return registers[idx % n_regs]
        
        def set_reg(idx, val):
            return registers.at[idx % n_regs].set(val)
        
        # READ_SIZE
        is_read_size = (op == READ_SIZE)
        registers = jnp.where(is_read_size & valid_op, set_reg(arg0, genome_len), registers)
        
        # ALLOCATE
        is_allocate = (op == ALLOCATE)
        alloc_size = get_reg(arg0)
        valid_alloc = (alloc_size > 0) & (alloc_size < cfg.max_genome_len)
        child_len = jnp.where(is_allocate & valid_op & valid_alloc, alloc_size, child_len)
        child_genome = jnp.where(
            is_allocate & valid_op & valid_alloc,
            jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32),
            child_genome
        )
        child_write_ptr = jnp.where(is_allocate & valid_op & valid_alloc, 0, child_write_ptr)
        
        # LOAD: registers[arg1] = genome[registers[arg0]]
        is_load = (op == LOAD)
        load_addr = get_reg(arg0)
        load_val = jnp.where((load_addr >= 0) & (load_addr < genome_len), genome[load_addr], 0)
        registers = jnp.where(is_load & valid_op, set_reg(arg1, load_val), registers)
        
        # STORE: child[registers[arg1]] = registers[arg0]
        is_store = (op == STORE)
        store_addr = get_reg(arg1)
        store_val = get_reg(arg0)
        valid_store = (child_len > 0) & (store_addr >= 0) & (store_addr < child_len)
        child_genome = jnp.where(
            is_store & valid_op & valid_store,
            child_genome.at[store_addr].set(store_val),
            child_genome
        )
        
        # MOVE: registers[arg1] = registers[arg0]
        is_move = (op == MOVE)
        registers = jnp.where(is_move & valid_op, set_reg(arg1, get_reg(arg0)), registers)
        
        # INC: registers[arg0] += 1
        is_inc = (op == INC)
        registers = jnp.where(is_inc & valid_op, set_reg(arg0, get_reg(arg0) + 1), registers)
        
        # DEC: registers[arg0] -= 1
        is_dec = (op == DEC)
        registers = jnp.where(is_dec & valid_op, set_reg(arg0, get_reg(arg0) - 1), registers)
        
        # ADD: registers[arg0] += registers[arg1]
        is_add = (op == ADD)
        registers = jnp.where(is_add & valid_op, set_reg(arg0, get_reg(arg0) + get_reg(arg1)), registers)
        
        # SUB: registers[arg0] -= registers[arg1]
        is_sub = (op == SUB)
        registers = jnp.where(is_sub & valid_op, set_reg(arg0, get_reg(arg0) - get_reg(arg1)), registers)
        
        # IFZERO: skip next if register is zero
        is_ifzero = (op == IFZERO)
        skip = (get_reg(arg0) == 0)
        ip = jnp.where(is_ifzero & valid_op & skip, ip + 1, ip)
        ip = jnp.where(ip >= genome_len, code_start, ip)
        
        # JUMP
        is_jump = (op == JUMP)
        jump_target = jnp.where(code_len > 0, code_start + (arg0 % code_len), code_start)
        ip = jnp.where(is_jump & valid_op, jump_target, ip)
        
        # DIVIDE: give birth
        is_divide = (op == DIVIDE)
        valid_divide = (child_len > 0)
        has_child = jnp.where(is_divide & valid_op & valid_divide, True, has_child)
        
        return (registers, child_genome, child_len, child_write_ptr, has_child, ip), None
    
    (registers, child_genome, child_len, child_write_ptr, has_child, ip), _ = lax.scan(
        exec_op,
        (registers, child_genome, child_len, child_write_ptr, has_child, ip),
        jnp.arange(cfg.max_ops_per_instr)
    )
    
    # Update state
    new_state = state.copy()
    new_state['registers'] = registers
    new_state['ip'] = ip
    new_state['child_genome'] = child_genome
    new_state['child_len'] = child_len
    new_state['child_write_ptr'] = child_write_ptr
    new_state['has_child'] = has_child
    
    return new_state


def run_organism_cycles(state: dict, cfg: Config):
    """Run an organism for cpu_cycles steps."""
    def step_fn(state, _):
        # Only step if alive and no child yet
        should_step = state['alive'] & ~state['has_child']
        new_state = lax.cond(
            should_step,
            lambda s: vm_step(s, cfg),
            lambda s: s,
            state
        )
        return new_state, None
    
    final_state, _ = lax.scan(step_fn, state, jnp.arange(cfg.cpu_cycles))
    return final_state


# ==========================================
# 6. MUTATION
# ==========================================

def mutate_genome(key: jax.Array, genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Apply mutations to a genome."""
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # Point mutation
    do_point = random.uniform(k1) < cfg.point_mutation_rate
    point_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    point_val = random.randint(k3, (), 0, 61)
    genome = jnp.where(
        do_point & (point_idx < genome_len),
        genome.at[point_idx].set(point_val),
        genome
    )
    
    # Indel mutation
    do_indel = random.uniform(k4) < cfg.indel_rate
    do_insert = random.uniform(k5) < 0.5
    
    # Insertion (shift right and insert)
    indel_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    insert_val = random.randint(k3, (), 0, 61)
    
    def do_insertion(args):
        genome, genome_len = args
        # Shift everything right from indel_idx
        new_genome = jnp.roll(genome, 1)
        # Fix: manually shift only from indel_idx onwards
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(indices > indel_idx, genome[indices - 1], genome[indices])
        shifted = shifted.at[indel_idx].set(insert_val)
        new_len = jnp.minimum(genome_len + 1, cfg.max_genome_len)
        return shifted, new_len
    
    # Deletion (shift left)
    def do_deletion(args):
        genome, genome_len = args
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(indices >= indel_idx, 
                           jnp.where(indices < cfg.max_genome_len - 1, genome[indices + 1], EMPTY),
                           genome[indices])
        new_len = jnp.maximum(genome_len - 1, 5)  # Minimum length 5
        return shifted, new_len
    
    genome, genome_len = lax.cond(
        do_indel,
        lambda args: lax.cond(do_insert & (args[1] < cfg.max_genome_len - 1), 
                              do_insertion, 
                              lambda a: lax.cond(args[1] > 5, do_deletion, lambda x: x, a),
                              args),
        lambda args: args,
        (genome, genome_len)
    )
    
    return genome, genome_len


# ==========================================
# 7. POPULATION DYNAMICS
# ==========================================

def create_ancestor_genome(cfg: Config):
    """Create the ancestor genome (dynamic mode)."""
    g = []
    # Hardware: 4 registers
    g += [R, R, R, R, B]
    
    # Instructions
    g += [I, READ_SIZE, 1]                           # I0: R1 = size
    g += [I, ALLOCATE, 1]                            # I1: allocate R1 bytes
    g += [I, LOAD, 2, 0, STORE, 0, 2, INC, 2]        # I2: copy loop
    g += [I, MOVE, 1, 3, SUB, 3, 2]                  # I3: R3 = R1 - R2
    g += [I, IFZERO, 3]                              # I4: skip if done
    g += [I, JUMP, 2]                                # I5: loop back
    g += [I, DIVIDE]                                 # I6: birth
    g += [SEP]
    
    # Code
    g += [0, 1, 2, 3, 4, 5, 6]
    
    genome = jnp.full(cfg.max_genome_len, EMPTY, dtype=jnp.int32)
    genome = genome.at[:len(g)].set(jnp.array(g, dtype=jnp.int32))
    genome_len = jnp.int32(len(g))
    
    return genome, genome_len


def init_organism(genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Initialize an organism from a genome."""
    state = create_organism_state(cfg)
    state['genome'] = genome
    state['genome_len'] = genome_len
    
    # Parse genome
    parsed = parse_genome(genome, genome_len, cfg)
    state['n_regs'] = parsed['n_regs']
    state['n_instructions'] = parsed['n_instructions']
    state['code_start'] = parsed['code_start']
    state['instr_ops'] = parsed['instr_ops']
    state['instr_args'] = parsed['instr_args']
    state['instr_n_ops'] = parsed['instr_n_ops']
    state['ip'] = parsed['code_start']
    
    return state


def init_population(key: jax.Array, cfg: Config):
    """Initialize population with ancestor genomes."""
    ancestor_genome, ancestor_len = create_ancestor_genome(cfg)
    
    # Vectorized init
    def init_one(i):
        return init_organism(ancestor_genome, ancestor_len, cfg)
    
    # Stack all organisms
    pop = jax.vmap(init_one)(jnp.arange(cfg.pop_size))
    return pop


# ==========================================
# 8. EPOCH STEP
# ==========================================

def epoch_step(carry, key):
    """Run one epoch of the simulation."""
    pop, cfg = carry
    k1, k2, k3 = random.split(key, 3)
    
    # Run all organisms in parallel
    run_fn = lambda state: run_organism_cycles(state, cfg)
    pop = jax.vmap(run_fn)(pop)
    
    # Age all organisms
    pop['age'] = pop['age'] + 1
    
    # Collect children and their parents
    has_child = pop['has_child']
    child_genomes = pop['child_genome']
    child_lens = pop['child_len']
    
    # Count births
    n_births = jnp.sum(has_child)
    
    # Mutate children
    child_keys = random.split(k1, cfg.pop_size)
    
    def mutate_one(args):
        key, genome, length, has = args
        new_genome, new_len = lax.cond(
            has,
            lambda g: mutate_genome(key, g[0], g[1], cfg),
            lambda g: (g[0], g[1]),
            (genome, length)
        )
        return new_genome, new_len
    
    mutated = jax.vmap(mutate_one)((child_keys, child_genomes, child_lens, has_child))
    mutated_genomes, mutated_lens = mutated
    
    # Death by age
    too_old = pop['age'] >= cfg.max_age
    pop['alive'] = pop['alive'] & ~too_old
    n_aged_deaths = jnp.sum(too_old & pop['alive'])
    
    # Replace dead organisms with children
    # Simple strategy: find dead slots, fill with children
    alive_mask = pop['alive']
    dead_indices = jnp.where(~alive_mask, jnp.arange(cfg.pop_size), cfg.pop_size)
    child_indices = jnp.where(has_child, jnp.arange(cfg.pop_size), cfg.pop_size)
    
    # Sort to get valid indices first
    dead_indices = jnp.sort(dead_indices)
    child_indices = jnp.sort(child_indices)
    
    # Match children to dead slots
    def assign_children(carry, i):
        pop, child_genomes, child_lens, dead_idx_ptr, child_idx_ptr = carry
        
        dead_slot = dead_indices[dead_idx_ptr]
        child_src = child_indices[child_idx_ptr]
        
        valid = (dead_slot < cfg.pop_size) & (child_src < cfg.pop_size)
        
        # Initialize child organism in dead slot
        def do_assign(args):
            pop, slot, genome, length = args
            new_state = init_organism(genome, length, cfg)
            # Update pop arrays at slot
            pop = jax.tree_map(lambda p, n: p.at[slot].set(n), pop, new_state)
            return pop
        
        pop = lax.cond(
            valid,
            do_assign,
            lambda args: args[0],
            (pop, dead_slot, mutated_genomes[child_src], mutated_lens[child_src])
        )
        
        dead_idx_ptr = jnp.where(valid, dead_idx_ptr + 1, dead_idx_ptr)
        child_idx_ptr = jnp.where(valid, child_idx_ptr + 1, child_idx_ptr)
        
        return (pop, child_genomes, child_lens, dead_idx_ptr, child_idx_ptr), None
    
    (pop, _, _, _, _), _ = lax.scan(
        assign_children,
        (pop, mutated_genomes, mutated_lens, jnp.int32(0), jnp.int32(0)),
        jnp.arange(cfg.pop_size)
    )
    
    # Reset child buffers
    pop['has_child'] = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    pop['child_len'] = jnp.zeros(cfg.pop_size, dtype=jnp.int32)
    
    # Compute stats
    alive_count = jnp.sum(pop['alive'])
    avg_genome_len = jnp.mean(jnp.where(pop['alive'], pop['genome_len'], 0))
    
    stats = {
        'pop_size': alive_count,
        'births': n_births,
        'avg_genome_len': avg_genome_len,
    }
    
    return (pop, cfg), stats


# ==========================================
# 9. MAIN SIMULATION
# ==========================================

def run_simulation(key: jax.Array, cfg: Config, epochs: int):
    """Run the full simulation."""
    print(f"=== JAX PHYSIS SIMULATION ===")
    print(f"Population: {cfg.pop_size}, Epochs: {epochs}, CPU cycles: {cfg.cpu_cycles}")
    print()
    
    # Initialize
    k1, k2 = random.split(key)
    pop = init_population(k1, cfg)
    
    # Run epochs
    epoch_keys = random.split(k2, epochs)
    
    # Use scan for epochs (but we'll do it in chunks for logging)
    chunk_size = 100
    n_chunks = epochs // chunk_size
    
    all_stats = []
    
    for chunk in range(n_chunks):
        chunk_keys = epoch_keys[chunk * chunk_size:(chunk + 1) * chunk_size]
        (pop, cfg), stats = lax.scan(epoch_step, (pop, cfg), chunk_keys)
        
        # Log last stats of chunk
        epoch_num = (chunk + 1) * chunk_size
        print(f"Epoch {epoch_num}: Pop={int(stats['pop_size'][-1])}, "
              f"Births={int(stats['births'][-1])}, "
              f"AvgLen={float(stats['avg_genome_len'][-1]):.1f}")
        
        all_stats.append(jax.tree_map(lambda x: np.array(x), stats))
    
    return pop, all_stats


# ==========================================
# 10. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Physis JAX - Parallel Digital Evolution')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--pop-size', type=int, default=1000, help='Population size')
    parser.add_argument('--cpu-cycles', type=int, default=500, help='CPU cycles per organism')
    parser.add_argument('--max-genome', type=int, default=100, help='Max genome length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    cfg = make_config(
        pop_size=args.pop_size,
        cpu_cycles=args.cpu_cycles,
        max_genome_len=args.max_genome,
    )
    
    key = random.PRNGKey(args.seed)
    
    # JIT compile the simulation
    print("JIT compiling...")
    run_simulation_jit = jax.jit(lambda k, c, e: run_simulation(k, c, e), static_argnums=(1, 2))
    
    # Run
    import time
    start = time.time()
    pop, stats = run_simulation(key, cfg, args.epochs)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Epochs/sec: {args.epochs / elapsed:.1f}")
