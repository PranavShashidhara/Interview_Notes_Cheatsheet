# HIP Kernel Fundamentals

## Execution Model
- Kernel = function running on GPU, launched by host (CPU)
- Kernel code is executed by many GPU threads **concurrently**
- Each thread does the identical operation on different data — **data parallelism**, not control flow parallelism
- C++ friendly model for expressing massive parallelism on modern GPUs
- GPUs apply the same operation via lightweight threads across thousands or millions of data elements simultaneously
- Parallel execution enables speedups for scientific, imaging, and simulation workloads
- Programming model splits work into small independent tasks with minimal coordination — foundation for more complex optimizations

## Host-Driven Control
- All execution decisions are visible to the developer
- Kernel launches, memory transfers, and synchronization points are controlled by the host
- HIP does not introduce implicit or hidden execution behavior
- Produces predictable, debuggable programs

## Programming Flow
```
CPU → Copy (Host→Device) → GPU runs kernel → Copy (Device→Host) → CPU validates/uses results
```

## Thread Hierarchy
```
Grid → Blocks → Threads
```
- Grid contains blocks, blocks contain threads
- Threads within a block execute concurrently on the GPU
- Runtime assigns identifiers to blocks and threads
- Kernels use these IDs to determine which portion of the data to process
- These identifiers are the basis for computing global indices and distributing work
- **Bug source**: incorrect assumptions about block/thread organization → overlapping work, missing elements, invalid memory access

## Global Indexing
Each thread computes a unique global index to determine exactly which data element it processes.

**Index components:**
- Each thread has an index within its block (`threadIdx`)
- Each block has an index within the grid (`blockIdx`)
- Combined for global calculations → unique thread mapping

### 1D Indexing
```
globalIdx = blockIdx.x * blockDim.x + threadIdx.x
```
- One thread per element execution approach
- Global index identifies which array element a thread processes
- Mapping ensures independent execution with no overlap
- Avoids unintended data access
- Linear access patterns are predictable and simpler to debug

### 2D Indexing
```
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```
- For matrices/images — threads mapped to (x,y) coordinates from block and thread IDs
- Parallel execution reflects the logical layout of the data
- Improves readability and correctness
- Bounds checks required on both axes
- Invalid memory access at borders are eliminated by guards

### 3D Indexing
- For volumes (scientific simulations, spatial models) with width, height, depth
- Each thread corresponds to one element within a 3D data volume
- 3 coordinates linearized for memory access
- Safety checks apply to all 3 dimensions — guarding is essential
- Safety requirements increase with dimensions
- Same defensive pattern is reused across 1D/2D/3D

## Bounds and Guards
- **Bounds**: valid range of data positions based on input sizes
- **Challenge**: kernel launches often create more threads than data elements
- Grid dimensions are often rounded up to simplify launch configuration
- Not all threads map to valid data within the bounds of the input
- Extra threads must be handled explicitly
- Allowing unrestricted execution leads to incorrect results or program instability

**Solution — Guards:**
- Check whether each thread corresponds to a valid data position before memory access or computation
- Ensure correctness and stability
- Apply uniformly across 1D, 2D, and 3D index models
- Provide safe and predictable GPU execution

## Common GPU Indexing Mistakes
| Bug | Effect |
|-----|--------|
| Missing or incorrect bounds checks | Invalid memory access, corrupted output |
| Incorrect linearization | Threads compute wrong memory location, access unintended data |
| Off-by-one errors | At data boundaries when grid sizes are rounded up or handling edge conditions |
| Configuration issues | Incorrect assumptions about data dimensions or grid sizes → portions of input skipped |

## Kernel Launch
- Host CPU launches GPU kernel using HIP runtime APIs
- Host specifies grid and block dimensions for thread creation
- Kernel execution begins **asynchronously** on the GPU
- Threads are created based on config and begin execution at the kernel's entry function

### Block and Grid Dimensions
- **Block size**: limited by hardware (how many threads per block — typically up to 1024). Consider portability
- **Grid size**: derived from total data to process. Must be large enough to cover all elements (may include extra threads)
- Grid dimensions: `ceil(N / blockSize)`

### Launch Configuration Template
Standard pattern:
1. Compute grid dimensions from data size and block size (conforming to hardware limits)
2. Perform error checks to catch configuration or runtime errors
3. Synchronization may be used during development to validate correctness

## Determinism & Initialization
- Deterministic behavior = same inputs produce same results
- Repeatable results simplify debugging and validation
- Non-repeatable results hide logical and indexing issues
- Establish determinism early — builds confidence before deeper validation or performance analysis
- **Uninitialized memory** is a common source of non-determinism
- If buffers are not explicitly initialized, threads may operate on undefined values
- Explicit initialization of inputs and output buffers removes ambiguity and ensures results reflect kernel logic

## Error Checking & Synchronization
- Kernel launches **fail silently** if errors are not checked
- Runtime APIs available for querying kernel launch status
- Check errors immediately after launch
- **Synchronization** forces completion of GPU work — required for debugging and validating results
- Avoid excessive synchronization — it can hide performance issues

## Validation
- Always validate GPU results against CPU reference **before** optimizing
- Optimizing incorrect code amplifies errors
- Use GPU references and tolerance-based comparisons to establish correctness

## Host vs Kernel Responsibilities
| Host (CPU) | Kernel (GPU) |
|------------|--------------|
| Kernel launch configuration | Computation only focus |
| Memory management | Each thread performs assigned work using provided indices |
| Execution flow orchestration | No management of memory movement or execution order |

## Event Timing
- Event timings are more accurate than host-side timers — intended for performance analysis, not correctness
- Use only after correctness is confirmed
- Timing is not a debugging substitute

## Debugging Checklist
1. Start with small input sizes to reason about thread behavior
2. Verify indexing and boundary conditions are correct
3. Add error checks and synchronization points to confirm launches and execution complete as expected
4. Compare GPU results with CPU output side-by-side to confirm correctness and diagnose discrepancies
5. Validate kernels step by step to simplify debugging and build confidence

## Common Patterns

### SAXPY (`a*X + Y`)
- Single precision element-wise computation
- Natural fit for 1D indexing — each thread performs same arithmetic on different array element
- Pattern: compute global index → apply bounds guard → perform computation on device-resident data

### 2D Stencil
- Each output value depends on a **neighborhood** of input values
- Thread computes one output but reads multiple neighboring inputs around its position
- Used in image processing and scientific computations

### Reduction
- Combines many input values into a single output value
- Requires threads to **work together** rather than independently
- Threads compute partial results and combine them via synchronization
- Common in real applications — foundation for more advanced algorithms
- Recognizing when a problem is a reduction helps with choosing the right strategy
