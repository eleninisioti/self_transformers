# Working Replicator Genomes

Use this file to track different versions of the replicator genome.

## 1. Efficient Replicator (V1)
**Description**: Uses a compact loop body where LOAD, STORE, and INC are executed in a single instruction cycle. Fast replication (1 cycle/gene). This was the initial stable version.

**Code**:
```python
g = []
# Hardware: 4 registers
g += [R, R, R, R, B]

# Instructions
g += [I, READ_SIZE, 1]                           # I0: R1 = size
g += [I, ALLOCATE, 1]                            # I1: allocate R1 bytes
g += [I, SUB, 2, 2]                              # I2: R2 = 0
g += [I, LOAD, 2, 0, STORE, 0, 2, INC, 2]        # I3: copy loop body (Compact)
g += [I, MOVE, 1, 3, SUB, 3, 2]                  # I4: R3 = R1 - R2
g += [I, IFZERO, 3, 1]                           # I5: skip next (JUMP) if done
g += [I, JUMP, 3]                                # I6: loop back to I3
g += [I, DIVIDE]                                 # I7: birth
g += [SEP]

# Data (Code indices)
g += [0, 1, 2, 3, 4, 5, 6]
```

## 2. Inefficient Replicator (V2 - Current)
**Description**: The copy operations are split into three separate instructions (`LOAD`, `STORE`, `INC`). This increases the replication cost to 3 cycles/gene (plus overhead). 
**Purpose**: Intentionally inefficient to allow "evolutionary headroom" for the population to discover compaction mutations (merging I3-I5 back into one instruction).

**Code**:
```python
g = []
# Hardware: 4 registers
g += [R, R, R, R, B]

# Instructions
g += [I, READ_SIZE, 1]                           # I0: R1 = size
g += [I, ALLOCATE, 1]                            # I1: allocate R1 bytes
g += [I, SUB, 2, 2]                              # I2: R2 = 0 (loop counter)

# Inefficient Loop Body (Split into 3 instructions)
g += [I, LOAD, 2, 0]                             # I3: R0 = genome[R2]
g += [I, STORE, 0, 2]                            # I4: child[R2] = R0
g += [I, INC, 2]                                 # I5: R2++

g += [I, MOVE, 1, 3, SUB, 3, 2]                  # I6: R3 = R1 - R2
g += [I, IFZERO, 3, 1]                           # I7: skip next (JUMP) if done
g += [I, JUMP, 3]                                # I8: loop back to I3
g += [I, DIVIDE]                                 # I9: birth
g += [SEP]

# Code (Indices updated for longer program)
g += [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
