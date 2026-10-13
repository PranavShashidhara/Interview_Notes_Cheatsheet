# Assembly Primer for Kernel Engineers

---

## Why Assembly Matters for This Role
- Understand what the compiler generates from your C++ kernels
- Identify missed vectorization, unnecessary loads, spills to stack
- NEON (ARM) is explicitly listed in the JD — edge devices run ARM

---

## 1. Registers (x86-64)

```
General purpose:   rax, rbx, rcx, rdx, rsi, rdi, rsp, rbp, r8-r15
Float/SIMD:        xmm0-xmm15   (128-bit, SSE)
                   ymm0-ymm15   (256-bit, AVX)
                   zmm0-zmm31   (512-bit, AVX-512)
```

- Registers are the fastest storage — no memory access at all
- `float a = A[i]` hoisted out of loop → compiler keeps `a` in a register
- If you have too many variables → **register spill** → stored to stack (slow)

---

## 2. Basic Instructions

```asm
mov  rax, rbx        ; rax = rbx
add  rax, 8          ; rax += 8
imul rax, rcx        ; rax *= rcx
lea  rax, [rbx + rcx*4]  ; rax = rbx + rcx*4  (address calc, no memory read)

; float ops
movss  xmm0, [rax]   ; load float from memory into xmm0
addss  xmm0, xmm1    ; xmm0 += xmm1 (scalar float add)
mulss  xmm0, xmm1    ; xmm0 *= xmm1 (scalar float mul)
```

---

## 3. SIMD — Doing Multiple Operations at Once

Instead of one float at a time, process 4/8 floats in one instruction:

```
Scalar (no SIMD):
xmm0 = [ a0 | -- | -- | -- ]   one float

SSE (128-bit):
xmm0 = [ a0 | a1 | a2 | a3 ]   4 floats at once

AVX (256-bit):
ymm0 = [ a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7 ]   8 floats at once
```

**SSE instructions:**
```asm
movaps  xmm0, [rax]      ; load 4 floats (aligned)
addps   xmm0, xmm1       ; add 4 floats at once  (ps = packed single)
mulps   xmm0, xmm1       ; mul 4 floats at once
```

**AVX instructions:**
```asm
vmovaps  ymm0, [rax]     ; load 8 floats
vaddps   ymm0, ymm1      ; add 8 floats at once
vmulps   ymm0, ymm1      ; mul 8 floats at once
vfmadd231ps ymm0, ymm1, ymm2  ; ymm0 += ymm1 * ymm2  (fused multiply-add)
```

**FMA (Fused Multiply-Add) — key for matmul:**
```asm
vfmadd231ps ymm0, ymm1, ymm2
; ymm0 = ymm0 + ymm1 * ymm2
; one instruction instead of mul + add
; also more numerically accurate
```

---

## 4. NEON (ARM) — What Edge Devices Use

```asm
; ARM NEON — 128-bit registers (v0-v31)
ld1  {v0.4s}, [x0]       ; load 4 floats from address in x0
fadd v0.4s, v0.4s, v1.4s ; add 4 floats
fmul v0.4s, v0.4s, v1.4s ; mul 4 floats
fmla v0.4s, v1.4s, v2.4s ; v0 += v1 * v2  (fused multiply-add)
st1  {v0.4s}, [x0]       ; store 4 floats
```

NEON is essentially ARM's version of SSE/AVX — same concept, different syntax.

---

## 5. How C++ Maps to Assembly

```cpp
// C++
float a = x[i] * y[i];
```
```asm
; scalar
movss xmm0, [rdi + rsi*4]   ; load x[i]
mulss xmm0, [rdx + rsi*4]   ; multiply y[i]
```

```cpp
// auto-vectorized by compiler
for (int i = 0; i < N; i++)
    c[i] = a[i] + b[i];
```
```asm
; compiler generates AVX
vmovaps ymm0, [rdi]        ; load 8 floats of a
vaddps  ymm0, ymm0, [rsi]  ; add 8 floats of b
vmovaps [rdx], ymm0        ; store to c
```

---

## 6. Memory Access Patterns in Assembly

```asm
; Sequential (good — cache line used fully)
mov rax, [rbx]        ; load from address rbx
mov rax, [rbx + 4]    ; next float, adjacent
mov rax, [rbx + 8]    ; next float, adjacent

; Strided (bad — cache line wasted)
mov rax, [rbx]        ; load
mov rax, [rbx + 256]  ; jump 64 floats forward, new cache line
mov rax, [rbx + 512]  ; jump again
```

---

## 7. What to Look for When Reading Assembly Output

Use `godbolt.org` to see what your C++ compiles to:

```
Good signs:
✓ ymm/zmm registers used        → compiler auto-vectorized
✓ vfmadd instructions           → fused multiply-add used
✓ loop unrolling visible        → multiple iterations per loop body

Bad signs:
✗ only xmm scalar instructions  → not vectorized
✗ many mov to/from [rsp]        → register spill to stack
✗ rep movsb instead of vmovaps  → unaligned or non-vectorized copy
```

---

## 8. Alignment Matters

SIMD loads are fastest when data is aligned to 32 bytes (AVX) or 16 bytes (SSE):

```cpp
// C++ — aligned allocation
float* a = (float*)aligned_alloc(32, N * sizeof(float));  // 32-byte aligned for AVX

// tell compiler data is aligned
__builtin_assume_aligned(a, 32);
```

```asm
vmovaps ymm0, [rax]    ; aligned load — fast
vmovups ymm0, [rax]    ; unaligned load — slightly slower
```

---

## 9. Subroutines (Functions in Assembly)

Functions are just `call` and `ret` with a calling convention:

```asm
; calling a function
call  myFunc       ; pushes return address onto stack, jumps to myFunc

; inside myFunc
push  rbp          ; save caller's base pointer
mov   rbp, rsp     ; set up new stack frame
sub   rsp, 16      ; allocate local variables on stack

; ... do work ...

pop   rbp          ; restore caller's base pointer
ret                ; pop return address from stack, jump back
```

**Calling convention (x86-64 Linux):**
```
Arguments passed in:   rdi, rsi, rdx, rcx, r8, r9  (first 6)
                       rest go on the stack
Return value:          rax (integer), xmm0 (float)
Caller saves:          rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11
Callee saves:          rbx, rbp, r12-r15  (must restore before ret)
```

So when you write:
```cpp
float saxpy(float a, float* x, float* y, int n)
//                   rdi       rsi       rdx  rcx
```
The compiler automatically maps args to registers per this convention.

**Stack frame layout:**
```
high address
[ caller's frame  ]
[ return address  ]  ← rsp after call
[ saved rbp       ]  ← rbp after push rbp
[ local variables ]  ← rsp after sub rsp, N
low address
```

---

## 10. Interrupts

Interrupts are signals that stop normal execution and jump to a handler:

**Types:**
```
Hardware interrupts:  external events — timer tick, keyboard, DMA complete
Software interrupts:  triggered by code — system calls, exceptions
Exceptions:          CPU-generated — divide by zero, page fault, invalid opcode
```

**Software interrupt — system call:**
```asm
; Linux x86-64 syscall
mov  rax, 1        ; syscall number (1 = write)
mov  rdi, 1        ; fd = stdout
mov  rsi, msg      ; buffer address
mov  rdx, 13       ; length
syscall            ; triggers interrupt → kernel takes over → returns to here
```

**Interrupt flow:**
```
normal code running
     ↓
interrupt fires (hardware or syscall)
     ↓
CPU saves registers + rip (instruction pointer) to stack
     ↓
jumps to Interrupt Descriptor Table (IDT) → finds handler
     ↓
handler runs (kernel mode for hardware interrupts)
     ↓
iret instruction → restores registers, resumes normal code
```

**Relevance to kernel engineering:**
- DMA completion interrupts — signal when memory transfer is done (relevant for NPU data movement)
- Timer interrupts — used for scheduling, profiling (perf counters)
- Page faults — triggered when accessing unmapped memory, kernel maps it in
- In embedded/edge devices you write interrupt handlers directly

---

## Key Takeaways for Interview

| Concept | What to say |
|---|---|
| SIMD | Process 4/8 floats per instruction instead of 1 |
| FMA | fused multiply-add — one instruction for `a += b*c` |
| NEON | ARM's SIMD — what runs on edge devices in this role |
| Auto-vectorization | Compiler does SIMD automatically if access is sequential |
| Register spill | Too many vars → compiler writes to stack → slow |
| Alignment | 32-byte aligned data → faster AVX loads |
| Godbolt | Tool to inspect what assembly your C++ generates |

---

## Likely Interview Questions on Assembly

**Your experience — always lead with this:**
> "I worked with 8051 assembly in an academic setting — arithmetic, external interrupts triggered by sensor readings to toggle an LED, and subroutines. It gave me a solid foundation for understanding how hardware events drive software execution asynchronously — the same pattern as DMA completion interrupts on an NPU."

---

**Q1: RISC vs CISC?**
```
RISC (ARM, 8051):
- Few simple fixed-size instructions
- Compiler does more work
- Lower power — good for edge devices
- ARM NEON is RISC → relevant to this role

CISC (x86):
- Many complex variable-size instructions
- Hardware does more work
- Higher power consumption
```

---

**Q2: What is the stack and how does it work?**
```
- LIFO structure in memory
- SP (stack pointer) moves down on push, up on pop
- Used for:
  → return addresses when calling subroutines
  → saving registers in ISR
  → local variables
- On 8051: CALL pushes PC, RET pops it back
```

---

**Q3: What is a calling convention?**
```
- Agreement between caller and callee on:
  → which registers hold arguments
  → which register holds return value
  → who saves/restores registers
- x86-64: args in rdi, rsi, rdx... return in rax
- 8051: no convention — manually use A, R0-R7
```

---

**Q4: What is SIMD and why does it matter for AI kernels?**
```
- Single Instruction Multiple Data
- Process 4/8 floats at once instead of one
- AVX (x86): ymm registers — 8 floats
- NEON (ARM): v registers — 4 floats
- Critical for matmul, convolution — same op on many elements
- Auto-vectorized by compiler if memory access is sequential
```

---

**Q5: Latency vs Throughput?**
```
Latency:    time for one operation to complete
Throughput: how many operations per second

SIMD improves throughput — same latency per op,
but 8 ops happen simultaneously → 8x throughput
```

---

**Q6: What happens during a context switch?**
```
- Save all registers + PC to memory (process control block)
- Load next process registers from memory
- Expensive — cache may be cold for new process
- Relevant: interrupts cause mini context switches into ISR
```

---

**Q7: What is DMA and why is it used?**
```
- Direct Memory Access — hardware moves data without CPU
- CPU free to compute while DMA transfers next batch
- Core pattern in NPU inference:
  DMA loads next layer's weights → CPU/NPU computes current layer
- Completion signaled via interrupt → your 8051 experience!
```

---

**Q8: What is pipelining?**
```
CPU executes multiple instructions simultaneously in stages:
fetch → decode → execute → writeback

Why sequential code is fast — pipeline stays full
Branch misprediction flushes pipeline — expensive
Why if statements in tight kernel loops hurt performance
```

---

**Q9: What is branch misprediction?**
```
- CPU guesses which branch to take, executes speculatively
- If wrong → pipeline flush → ~15 cycles wasted
- Why bounds checks (if i < N) in tight loops matter
- Solution: branchless code where possible
  e.g. fmaxf(0, x) instead of if (x < 0) x = 0
```

---

**Q10: Walk me through matmul at hardware level**
```
1. Compiler generates NEON/AVX SIMD instructions
2. Instructions fetched into CPU pipeline
3. Data loaded: DRAM → L3 → L2 → L1 cache → registers
4. FMA executes on SIMD units (8 floats at once)
5. Result written back to memory
6. DMA handles weight transfer between CPU and NPU
7. DMA completion interrupt signals next kernel to start
```
This ties together: assembly + cache + SIMD + interrupts + kernels
