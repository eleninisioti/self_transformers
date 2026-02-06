"""
Physis Minimal - JAX Version

Parallelized digital evolution simulation using JAX.
- Population runs in parallel via vmap
- Cycles via lax.scan with interleaved birth handling
- Fixed-size padded genomes for JAX compatibility
"""
import os
# CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
from functools import partial
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import io
try:
    import imageio
except ImportError:
    imageio = None
from PIL import Image


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==========================================
# 1. CONSTANTS (Gene values)
# ==========================================

R = 100
S = 101
B = 102
I = 103
SEP = 104

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

NOP = -1
EMPTY = -1


# ==========================================
# 2. CONFIGURATION
# ==========================================

class Config:
    """Simulation configuration with fixed sizes for JAX."""
    max_genome_len: int = 128
    max_registers: int = 8
    max_instructions: int = 32
    max_ops_per_instr: int = 16
    pop_size: int = 1024
    initial_pop: int = 16
    max_age: int = 40000  # In cycles
    
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
        'instr_ops': jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32),
        'instr_args': jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32),
        'instr_n_ops': jnp.zeros(cfg.max_instructions, dtype=jnp.int32),
    }


# ==========================================
# 4. GENOME PARSING
# ==========================================

def parse_genome(genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Parse genome into phenotype using vectorized operations."""
    
    # Create position indices
    positions = jnp.arange(cfg.max_genome_len)
    valid_pos = positions < genome_len
    
    # A. Count registers: Rs at start before B, I, or SEP
    is_R = (genome == R) & valid_pos
    is_marker = ((genome == B) | (genome == I) | (genome == SEP)) & valid_pos
    
    # Find first marker position
    marker_positions = jnp.where(is_marker, positions, cfg.max_genome_len)
    first_marker = jnp.min(marker_positions)
    
    # Count Rs before first marker
    n_regs = jnp.sum(is_R & (positions < first_marker))
    n_regs = jnp.maximum(n_regs, 1)
    
    # Language section starts after first marker (skip B if present)
    language_start = jnp.where(genome[first_marker] == B, first_marker + 1, first_marker)
    language_start = jnp.minimum(language_start, genome_len)
    
    # B. Find SEP position (end of instruction definitions)
    is_SEP = (genome == SEP) & valid_pos & (positions >= language_start)
    sep_positions = jnp.where(is_SEP, positions, cfg.max_genome_len)
    sep_pos = jnp.min(sep_positions)
    
    # C. Find instruction markers (I) between language_start and sep_pos
    is_I = (genome == I) & valid_pos & (positions >= language_start) & (positions < sep_pos)
    
    # Get positions of I markers
    I_positions = jnp.where(is_I, positions, cfg.max_genome_len)
    I_positions = jnp.sort(I_positions)  # Valid positions first
    
    # D. Parse each instruction definition
    # For simplicity, use a scan but only over max_instructions (small, e.g. 20)
    instr_ops = jnp.full((cfg.max_instructions, cfg.max_ops_per_instr), NOP, dtype=jnp.int32)
    instr_args = jnp.zeros((cfg.max_instructions, cfg.max_ops_per_instr, 2), dtype=jnp.int32)
    instr_n_ops = jnp.zeros(cfg.max_instructions, dtype=jnp.int32)
    
    def parse_one_instruction(instr_idx):
        """Parse instruction at I_positions[instr_idx]."""
        start = I_positions[instr_idx]
        valid_instr = start < cfg.max_genome_len
        
        # Find end of this instruction (next I or SEP)
        next_markers = jnp.where(
            ((genome == I) | (genome == SEP)) & (positions > start) & valid_pos,
            positions,
            cfg.max_genome_len
        )
        end = jnp.min(next_markers)
        
        # Parse ops within this instruction
        ops_row = jnp.full(cfg.max_ops_per_instr, NOP, dtype=jnp.int32)
        args_row = jnp.zeros((cfg.max_ops_per_instr, 2), dtype=jnp.int32)
        
        def parse_op(carry, op_idx):
            ptr, ops_row, args_row, n_ops = carry
            
            at_end = (ptr >= end) | (ptr >= genome_len)
            gene = jnp.where(at_end, NOP, genome[ptr])
            
            is_2arg = (gene == MOVE) | (gene == LOAD) | (gene == STORE) | (gene == ADD) | (gene == SUB)
            is_1arg = (gene == READ_SIZE) | (gene == ALLOCATE) | (gene == INC) | (gene == DEC) | (gene == JUMP) | (gene == IFZERO)
            is_0arg = gene == DIVIDE
            is_op = is_2arg | is_1arg | is_0arg
            
            valid_op = is_op & ~at_end & (op_idx < cfg.max_ops_per_instr)
            
            ops_row = jnp.where(valid_op, ops_row.at[op_idx].set(gene), ops_row)
            
            arg0 = jnp.where((ptr + 1 < genome_len) & (is_1arg | is_2arg), genome[ptr + 1], 0)
            arg1 = jnp.where((ptr + 2 < genome_len) & is_2arg, genome[ptr + 2], 0)
            args_row = jnp.where(valid_op, args_row.at[op_idx, 0].set(arg0), args_row)
            args_row = jnp.where(valid_op, args_row.at[op_idx, 1].set(arg1), args_row)
            
            advance = jnp.where(is_2arg, 3, jnp.where(is_1arg, 2, jnp.where(is_0arg, 1, 1)))
            ptr = jnp.where(valid_op, ptr + advance, jnp.where(~at_end, ptr + 1, ptr))
            n_ops = jnp.where(valid_op, n_ops + 1, n_ops)
            
            return (ptr, ops_row, args_row, n_ops), None
        
        # Start parsing after the I marker
        init_ptr = jnp.where(valid_instr, start + 1, cfg.max_genome_len)
        (_, ops_row, args_row, n_ops), _ = lax.scan(
            parse_op,
            (init_ptr, ops_row, args_row, jnp.int32(0)),
            jnp.arange(cfg.max_ops_per_instr)
        )
        
        return ops_row, args_row, n_ops, valid_instr
    
    # Parse all instructions (vmap over instruction indices)
    all_ops, all_args, all_n_ops, all_valid = jax.vmap(parse_one_instruction)(
        jnp.arange(cfg.max_instructions)
    )
    
    # Only keep valid instructions
    instr_ops = jnp.where(all_valid[:, None], all_ops, instr_ops)
    instr_args = jnp.where(all_valid[:, None, None], all_args, instr_args)
    instr_n_ops = jnp.where(all_valid, all_n_ops, instr_n_ops)
    
    n_instructions = jnp.sum(all_valid.astype(jnp.int32))
    n_instructions = jnp.maximum(n_instructions, 1)
    
    # Code section starts after SEP
    code_start = jnp.where(sep_pos < genome_len, sep_pos + 1, genome_len)
    
    return {
        'n_regs': n_regs,
        'n_instructions': n_instructions,
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
    has_child = state['has_child']
    
    code_len = genome_len - code_start
    valid_code = code_len > 0
    
    ip = jnp.where(
        valid_code & ((ip < code_start) | (ip >= genome_len)),
        code_start,
        ip
    )
    
    op_code = jnp.where(valid_code & (ip < genome_len), genome[ip], 0)
    ip = ip + 1
    ip = jnp.where(ip >= genome_len, code_start, ip)
    
    instr_idx = jnp.where(n_instructions > 0, op_code % n_instructions, 0)
    
    def exec_op(carry, op_idx):
        registers, child_genome, child_len, has_child, ip = carry
        
        op = instr_ops[instr_idx, op_idx]
        arg0 = instr_args[instr_idx, op_idx, 0]
        arg1 = instr_args[instr_idx, op_idx, 1]
        
        valid_op = op != NOP
        
        def get_reg(idx):
            return registers[idx % n_regs]
        
        def set_reg(idx, val):
            return registers.at[idx % n_regs].set(val)
        
        # READ_SIZE
        registers = jnp.where(
            (op == READ_SIZE) & valid_op, 
            set_reg(arg0, genome_len), 
            registers
        )
        
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
        
        # LOAD: registers[arg1] = genome[registers[arg0]]
        is_load = op == LOAD
        load_addr = get_reg(arg0)
        load_val = jnp.where((load_addr >= 0) & (load_addr < genome_len), genome[load_addr], 0)
        registers = jnp.where(is_load & valid_op, set_reg(arg1, load_val), registers)
        
        # STORE: child[registers[arg1]] = registers[arg0]
        is_store = op == STORE
        store_addr = get_reg(arg1)
        store_val = get_reg(arg0)
        valid_store = (child_len > 0) & (store_addr >= 0) & (store_addr < child_len)
        child_genome = jnp.where(
            is_store & valid_op & valid_store,
            child_genome.at[store_addr].set(store_val),
            child_genome
        )
        
        # MOVE: registers[arg1] = registers[arg0]
        registers = jnp.where(
            (op == MOVE) & valid_op, 
            set_reg(arg1, get_reg(arg0)), 
            registers
        )
        
        # INC: registers[arg0] += 1
        registers = jnp.where(
            (op == INC) & valid_op, 
            set_reg(arg0, get_reg(arg0) + 1), 
            registers
        )
        
        # DEC: registers[arg0] -= 1
        registers = jnp.where(
            (op == DEC) & valid_op, 
            set_reg(arg0, get_reg(arg0) - 1), 
            registers
        )
        
        # ADD: registers[arg0] += registers[arg1]
        registers = jnp.where(
            (op == ADD) & valid_op, 
            set_reg(arg0, get_reg(arg0) + get_reg(arg1)), 
            registers
        )
        
        # SUB: registers[arg0] -= registers[arg1]
        registers = jnp.where(
            (op == SUB) & valid_op, 
            set_reg(arg0, get_reg(arg0) - get_reg(arg1)), 
            registers
        )
        
        # IFZERO: skip next if register is zero
        is_ifzero = op == IFZERO
        skip = get_reg(arg0) == 0
        ip = jnp.where(is_ifzero & valid_op & skip, ip + 1, ip)
        ip = jnp.where(ip >= genome_len, code_start, ip)
        
        # JUMP
        is_jump = op == JUMP
        jump_target = jnp.where(code_len > 0, code_start + (arg0 % code_len), code_start)
        ip = jnp.where(is_jump & valid_op, jump_target, ip)
        
        # DIVIDE: give birth
        is_divide = op == DIVIDE
        valid_divide = child_len > 0
        has_child = jnp.where(is_divide & valid_op & valid_divide, True, has_child)
        
        return (registers, child_genome, child_len, has_child, ip), None
    
    (registers, child_genome, child_len, has_child, ip), _ = lax.scan(
        exec_op,
        (registers, child_genome, child_len, has_child, ip),
        jnp.arange(cfg.max_ops_per_instr)
    )
    
    new_state = state.copy()
    new_state['registers'] = registers
    new_state['ip'] = ip
    new_state['child_genome'] = child_genome
    new_state['child_len'] = child_len
    new_state['has_child'] = has_child
    
    return new_state


# ==========================================
# 6. MUTATION
# ==========================================

def mutate_genome(key: jax.Array, genome: jnp.ndarray, genome_len: jnp.int32, cfg: Config):
    """Apply mutations to a genome."""
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # Point mutation
    do_point = random.uniform(k1) < cfg.point_mutation_rate
    point_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    point_val = random.randint(k3, (), 0, 128) # Updated range
    genome = jnp.where(
        do_point & (point_idx < genome_len),
        genome.at[point_idx].set(point_val),
        genome
    )
    
    # Indel mutation
    do_indel = random.uniform(k4) < cfg.indel_rate
    do_insert = random.uniform(k5) < 0.5
    indel_idx = random.randint(k2, (), 0, jnp.maximum(genome_len, 1))
    insert_val = random.randint(k3, (), 0, 128) # Updated range
    
    def do_insertion(args):
        genome, genome_len = args
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(indices > indel_idx, genome[indices - 1], genome[indices])
        shifted = shifted.at[indel_idx].set(insert_val)
        new_len = jnp.minimum(genome_len + 1, cfg.max_genome_len)
        return shifted, new_len
    
    def do_deletion(args):
        genome, genome_len = args
        indices = jnp.arange(cfg.max_genome_len)
        shifted = jnp.where(
            indices >= indel_idx, 
            jnp.where(indices < cfg.max_genome_len - 1, genome[indices + 1], EMPTY),
            genome[indices]
        )
        new_len = jnp.maximum(genome_len - 1, 5)
        return shifted, new_len
    
    genome, genome_len = lax.cond(
        do_indel,
        lambda args: lax.cond(
            do_insert & (args[1] < cfg.max_genome_len - 1), 
            do_insertion, 
            lambda a: lax.cond(args[1] > 5, do_deletion, lambda x: x, a),
            args
        ),
        lambda args: args,
        (genome, genome_len)
    )
    
    return genome, genome_len


# ==========================================
# 7. POPULATION INITIALIZATION
# ==========================================

def create_ancestor_genome(cfg: Config):
    """Create the ancestor genome."""
    g = []
    # Hardware: 4 registers
    g += [R, R, R, R, B]
    
    # Instructions
    g += [I, READ_SIZE, 1]                           # I0: R1 = size
    g += [I, ALLOCATE, 1]                            # I1: allocate R1 bytes
    g += [I, LOAD, 2, 0, STORE, 0, 2, INC, 2]        # I2: copy loop body
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
    """Initialize population with ancestor genomes at random positions."""
    ancestor_genome, ancestor_len = create_ancestor_genome(cfg)
    
    # Select random indices for initial population
    # We use random.choice without replacement logic (argsort shuffle)
    # to ensure unique positions
    k1, k2 = random.split(key)
    perm = random.permutation(k1, cfg.pop_size)
    # The first 'initial_pop' indices in the permutation are the alive ones
    alive_indices = perm[:cfg.initial_pop]
    
    # Create mask - tricky in JAX to be efficient?
    # Actually, we can just say: if index i is in alive_indices, then alive=True.
    # But checking "in" is O(N*M).
    # Better: create bool array.
    is_alive = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    is_alive = is_alive.at[alive_indices].set(True)
    
    def init_one(i, alive_mask_val):
        state = init_organism(ancestor_genome, ancestor_len, cfg)
        state['alive'] = alive_mask_val
        return state
    
    pop = jax.vmap(init_one)(jnp.arange(cfg.pop_size), is_alive)
    return pop


# ==========================================
# 8. CYCLE STEP (single cycle for all organisms)
# ==========================================

def cycle_step(cfg: Config, pop: dict, key: jax.Array):
    """Execute one cycle: step all organisms, handle births."""
    
    # Step all alive organisms that haven't reproduced yet
    def step_one(state):
        should_step = state['alive'] & ~state['has_child']
        return lax.cond(
            should_step,
            lambda s: vm_step(s, cfg),
            lambda s: s,
            state
        )
    
    pop = jax.vmap(step_one)(pop)
    
    # Age all alive organisms
    pop['age'] = jnp.where(pop['alive'], pop['age'] + 1, pop['age'])
    
    # Death by age
    too_old = pop['age'] >= cfg.max_age
    pop['alive'] = pop['alive'] & ~too_old
    
    # Collect births
    has_child = pop['has_child']
    n_births = jnp.sum(has_child)
    
    # Mutate children
    mut_keys = random.split(key, cfg.pop_size)
    
    def mutate_one(args):
        key, genome, length, has = args
        return lax.cond(
            has,
            lambda g: mutate_genome(key, g[0], g[1], cfg),
            lambda g: (g[0], g[1]),
            (genome, length)
        )
    
    mutated_genomes, mutated_lens = jax.vmap(mutate_one)(
        (mut_keys, pop['child_genome'], pop['child_len'], has_child)
    )
    

    # Spatial Reproduction (Grid Physics)
    # Treat 1D array as (H, W) grid
    # For simplicity, let's assume square grid or close to it
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    # Ensure pop_size maps to grid exactly? 
    # Current code allows any pop_size. For correct wrapping, we ideally want perfect square.
    # But we can just use (H, W) where W = grid_side
    W = grid_side
    H = (cfg.pop_size + W - 1) // W # Ceiling division
    
    # Indices 0..pop_size-1
    # Check parents
    parent_indices = jnp.arange(cfg.pop_size)
    
    # We only care about parents who have a child
    # has_child is mask
    
    # For EACH organism (whether parent or not, we can compute neighbors vectorized)
    # Let's compute the "target slot" for every organism assuming it IS a parent
    # Then mask by has_child
    
    # 1. Get neighbors indices (8 neighbors)
    y = parent_indices // W
    x = parent_indices % W
    
    dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dx = jnp.array([-1, 0, 1, -1, 1, -1, 0, 1])
    
    # Broadcast to (pop_size, 8)
    ny = (y[:, None] + dy[None, :]) % H
    nx = (x[:, None] + dx[None, :]) % W
    
    neighbor_indices = ny * W + nx
    
    # Wrap handling: indices might be >= pop_size if pop_size isn't perfect rectangle/square fill
    # Mask invalid neighbors (those beyond pop_size)
    neighbor_valid = neighbor_indices < cfg.pop_size
    # If invalid, point to self or 0 (we will mask score anyway)
    neighbor_indices_safe = jnp.where(neighbor_valid, neighbor_indices, parent_indices[:, None])
    
    # 2. Get neighbor states
    # We need: is_empty (not alive), age
    neighbor_alive = pop['alive'][neighbor_indices_safe]
    neighbor_age = pop['age'][neighbor_indices_safe]
    
    # 3. Score neighbors
    # Priority 1: Empty (not alive). Score = Infinity (or very high)
    # Priority 2: Oldest. Score = age
    
    # If neighbor invalid index, score = -1 (never pick)
    # If neighbor empty: score = 1e9
    # If neighbor alive: score = age
    
    base_score = jnp.where(neighbor_alive, neighbor_age.astype(jnp.float32), 1e9)
    final_score = jnp.where(neighbor_valid, base_score, -1.0)
    
    # 4. Select best target for each parent
    # argmax gives index 0..7
    best_neighbor_local_idx = jnp.argmax(final_score, axis=1)
    
    # Gather the global index corresponding to that local idx
    # neighbor_indices_safe is (N, 8)
    # best_neighbor_local_idx is (N,)
    target_indices = jnp.take_along_axis(neighbor_indices_safe, best_neighbor_local_idx[:, None], axis=1).squeeze(1)
    
    # 5. Prepare updates
    # Only actual parents generate updates
    # We have `mutated_genomes` and `mutated_lens` aligned with parents (0..N-1, valid if has_child)
    
    # Parse children genomes (aligned with parents)
    child_parsed = jax.vmap(lambda g, l: parse_genome(g, l, cfg))(mutated_genomes, mutated_lens)
    
    # Build child states
    def build_child_state(genome, genome_len, parsed):
        state = create_organism_state(cfg)
        state['genome'] = genome
        state['genome_len'] = genome_len
        state['n_regs'] = parsed['n_regs']
        state['n_instructions'] = parsed['n_instructions']
        state['code_start'] = parsed['code_start']
        state['instr_ops'] = parsed['instr_ops']
        state['instr_args'] = parsed['instr_args']
        state['instr_n_ops'] = parsed['instr_n_ops']
        state['ip'] = parsed['code_start']
        return state
        
    child_states = jax.vmap(build_child_state)(mutated_genomes, mutated_lens, child_parsed)
    
    # 6. Scatter updates
    # We want to place `child_states` at `target_indices` where `has_child` is True.
    # To handle the mask efficiently in JAX scatter, we map inputs such that:
    # - If has_child: update = child_state (writes child to target)
    # - If !has_child: update = pop (writes parent to self, effectively no-op)
    
    real_target_indices = jnp.where(has_child, target_indices, parent_indices)

    # Better approach: map over (child_states, pop)
    update_payload = jax.tree.map(
        lambda c, p: jnp.where(
            has_child.reshape((-1,) + (1,) * (c.ndim - 1)),
            c,
            p
        ),
        child_states,
        pop
    )
    
    # Now scatter
    def scatter_update(current, update):
        return current.at[real_target_indices].set(update)
        
    pop = jax.tree.map(lambda c, u: scatter_update(c, u), pop, update_payload)
    
    # Reset child buffers
    pop['has_child'] = jnp.zeros(cfg.pop_size, dtype=jnp.bool_)
    pop['child_len'] = jnp.where(has_child, jnp.int32(0), pop['child_len'])
    
    # Stats
    alive_count = jnp.sum(pop['alive'])
    avg_genome_len = jnp.sum(jnp.where(pop['alive'], pop['genome_len'], 0)) / jnp.maximum(alive_count, 1)
    
    stats = {
        'pop_size': alive_count,
        'births': n_births,
        'avg_genome_len': avg_genome_len,
    }
    
    return pop, stats


# ==========================================
# 9. VISUALIZATION
# ==========================================

def plot_metrics(timestamps, pop_sizes, avg_lens, filename="metrics.png"):
    """Plot population size and average genome length over time."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Population Size', color=color)
    ax1.plot(timestamps, pop_sizes, color=color, label='Pop Size')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Avg Genome Length', color=color)  # we already handled the x-label with ax1
    ax2.plot(timestamps, avg_lens, color=color, linestyle='--', label='Avg Len')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Simulation Metrics')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(filename)
    plt.close()
    print(f"Saved metrics plot to {filename}")


def save_grid_gif(snapshots, filename, cfg):
    """Generate a GIF of the 2D grid representation."""
    print("Generating GIF...")
    frames = []
    
    # Calculate grid dimensions (approx square)
    grid_side = int(np.ceil(np.sqrt(cfg.pop_size)))
    
    # Find max genome len for normalization
    max_len = cfg.max_genome_len
    
    for i, snap in enumerate(snapshots):
        alive_mask = snap['alive']
        genome_lens = snap['genome_len']
        
        # Create grid images
        # 1. Filter metrics to grid size (pad if needed)
        pad_size = grid_side * grid_side - cfg.pop_size
        
        # Prepare data
        alive_grid = np.pad(alive_mask, (0, pad_size), constant_values=False).reshape(grid_side, grid_side)
        len_grid = np.pad(genome_lens, (0, pad_size), constant_values=0).reshape(grid_side, grid_side)
        
        # Create RGB image
        # Background: Black (0,0,0) or Very Dark Gray
        # Alive: Colored by length (Blue -> Red via Viridis or similar)
        
        # Normalize length for color mapping (0 to max_genome_len)
        norm_len = np.clip(len_grid / max_len, 0, 1)
        
        # Use a colormap
        cmap = plt.get_cmap('viridis')
        rgba = cmap(norm_len)  # (N, N, 4)
        
        # Apply mask: Dead cells are black
        # alpha is 1 where alive, 0 where dead (visual distinction)
        # But for 'black' dead cells, we just zero out the RGB
        
        rgb = rgba[..., :3] # (N, N, 3)
        
        # Vectorized masking
        mask = alive_grid[..., None]
        final_img = np.where(mask, rgb, 0.0) # Black background for dead
        
        # Upscale for visibility?
        # Let's keep 1 pixel per cell for now, maybe use matplotlib to save frame
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(final_img, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"Cycle {snap['cycle']}")
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        frames.append(imageio.imread(buf) if imageio else np.array(Image.open(buf)))
        
    if imageio:
        imageio.mimsave(filename, frames, fps=10)
        print(f"Saved GIF to {filename}")
    else:
        print("imageio not installed, saving frames as separate images not implemented for simplicity.")


# ==========================================
# 10. MAIN SIMULATION
# ==========================================

def run_simulation(key: jax.Array, cfg: Config, total_cycles: int, 
                   log_interval: int = 10000, use_wandb: bool = False):
    """Run the simulation for total_cycles."""
    print(f"=== JAX PHYSIS SIMULATION ===")
    print(f"Population capacity: {cfg.pop_size}, Initial: {cfg.initial_pop}")
    print(f"Total cycles: {total_cycles}, Log interval: {log_interval}")
    print()
    
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed. Run: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project="physis-jax",
                config={
                    "total_cycles": total_cycles,
                    "pop_size": cfg.pop_size,
                    "initial_pop": cfg.initial_pop,
                    "max_genome_len": cfg.max_genome_len,
                    "max_age": cfg.max_age,
                    "point_mutation_rate": cfg.point_mutation_rate,
                    "indel_rate": cfg.indel_rate,
                }
            )
    
    # Initialize
    k1, k2 = random.split(key)
    pop = init_population(k1, cfg)
    
    # JIT compile the cycle step
    cycle_step_fn = partial(cycle_step, cfg)
    
    def scan_cycles(pop, keys):
        def step(pop, key):
            pop, stats = cycle_step_fn(pop, key)
            return pop, stats
        return lax.scan(step, pop, keys)
    
    jit_scan = jax.jit(scan_cycles)
    
    # Run in chunks for logging
    n_chunks = total_cycles // log_interval
    all_stats = []
    
    cycle_keys = random.split(k2, total_cycles)
    
    for chunk in trange(n_chunks, desc="Running"):
        start = chunk * log_interval
        end = (chunk + 1) * log_interval
        chunk_keys = cycle_keys[start:end]
        
        pop, stats = jit_scan(pop, chunk_keys)
        # Block until computed
        pop = jax.block_until_ready(pop)
        
        # Log last stats of chunk
        cycle_num = end
        pop_size = int(stats['pop_size'][-1])
        births = int(jnp.sum(stats['births']))
        avg_len = float(stats['avg_genome_len'][-1])
        
        print(f"Cycle {cycle_num}: Pop={pop_size}, Births={births}, AvgLen={avg_len:.1f}")
        
        if use_wandb:
            wandb.log({
                "cycle": cycle_num,
                "population/size": pop_size,
                "population/births_interval": births,
                "genome/avg_len": avg_len,
            })
        
        # Collect snapshot for this chunk (end state)
        snapshot = {
            'cycle': cycle_num,
            'alive': np.array(pop['alive']),
            'genome_len': np.array(pop['genome_len'])
        }
        
        chunk_rec = {
            'cycle': cycle_num,
            'pop_size': pop_size,
            'births': births,
            'avg_len': avg_len,
            'snapshot': snapshot
        }
        
        all_stats.append(chunk_rec)
    
    if use_wandb:
        wandb.finish()
    
    return pop, all_stats


# ==========================================
# 11. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    cfg = make_config(
        pop_size=1024,
        initial_pop=4,
        max_age=40_000,
        point_mutation_rate=0.02,
        indel_rate=0.01,
    )

    key = random.PRNGKey(42)
    pop, stats = run_simulation(
        key, 
        cfg, 
        total_cycles=10000,
        log_interval=100,
        use_wandb=False,
    )

    print("\n=== FINAL STATE ===")
    alive = pop['alive']

    print(f"Avg genome length: {float(jnp.mean(jnp.where(alive, pop['genome_len'], 0))):.1f}")

    # Plotting and GIF generation
    timestamps = [s['cycle'] for s in stats]
    pop_sizes = [s['pop_size'] for s in stats]
    avg_lens = [s['avg_len'] for s in stats]
    
    # 1. Plot metrics
    plot_metrics(timestamps, pop_sizes, avg_lens, "simulation_metrics.png")
    
    # 2. Generate GIF
    # We collected snapshots in stats['snapshots'] (added in run_simulation)
    # Extract snapshots
    snapshots = [s['snapshot'] for s in stats]
    save_grid_gif(snapshots, "evolution.gif", cfg)