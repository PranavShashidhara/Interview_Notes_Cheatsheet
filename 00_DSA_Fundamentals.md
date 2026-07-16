# Data Structures & Algorithms (DSA) Guide

Comprehensive reference for technical interview preparation covering essential data structures, algorithms, complexity analysis, and problem-solving patterns.

---

## Table of Contents
1. [Complexity Analysis](#complexity-analysis)
2. [Data Structures](#data-structures)
3. [Algorithms](#algorithms)
4. [Problem-Solving Patterns](#problem-solving-patterns)
5. [Common Interview Questions](#common-interview-questions)

---

## Complexity Analysis

### Time Complexity

**Big O Notation Rankings (Best to Worst):**
- O(1) - Constant
- O(log n) - Logarithmic (binary search)
- O(n) - Linear (single loop)
- O(n log n) - Linearithmic (merge sort, quick sort)
- O(n²) - Quadratic (nested loops, bubble sort)
- O(n³) - Cubic (triple nested loops)
- O(2ⁿ) - Exponential (recursive subsets)
- O(n!) - Factorial (permutations)

**Key Points:**
- Focus on the dominant term
- Drop constants: O(2n) = O(n)
- Nested loops multiply: O(n * m)
- Sequential loops add: O(n + m)

### Space Complexity
- Consider auxiliary space (not input size)
- Recursive call stack counts
- Data structure allocation
- **Auxiliary Space**: extra memory used besides input

---

## Data Structures

### 1. Arrays & Strings

**Definition:** Contiguous memory blocks with fixed/dynamic size

**Operations:**
- Access: O(1)
- Insert/Delete: O(n)
- Search: O(n) linear, O(log n) sorted

**Common Patterns:**
- Two pointers
- Sliding window
- Prefix/Suffix sum
- Binary search

**Use Cases:**
- Direct indexing needed
- Cache-friendly access
- Space-efficient storage

---

### 2. Linked Lists

**Definition:** Chain of nodes with data and next pointer

**Variations:**
- Singly Linked List
- Doubly Linked List (bidirectional traversal)
- Circular Linked List

**Operations:**
- Access: O(n)
- Insert/Delete: O(1) if pointer given, O(n) to find
- Search: O(n)

**Key Techniques:**
- Fast/slow pointers (cycle detection)
- Reverse list
- Merge sorted lists
- Partition list

**Advantages:**
- Dynamic size
- Efficient insertions/deletions at known positions

**Disadvantages:**
- No random access
- Extra space for pointers
- Cache unfriendly

---

### 3. Stacks (LIFO)

**Definition:** Last In First Out - access from top only

**Operations:** O(1)
- Push: add to top
- Pop: remove from top
- Peek: view top element

**Use Cases:**
- Function call stack
- Undo/Redo functionality
- Expression evaluation
- Backtracking problems
- DFS (Depth-First Search)

**Classic Problems:**
- Balanced parentheses
- Next greater element
- Largest rectangle in histogram
- Daily temperatures

---

### 4. Queues (FIFO)

**Definition:** First In First Out - add at rear, remove from front

**Variations:**
- Simple Queue
- Circular Queue (optimal space)
- Deque (double-ended queue)
- Priority Queue

**Operations:** O(1)
- Enqueue: add to rear
- Dequeue: remove from front
- Peek: view front element

**Use Cases:**
- BFS (Breadth-First Search)
- Level-order traversal
- Task scheduling
- Print queue systems

---

### 5. Hash Tables / Hash Maps

**Definition:** Key-value pairs using hash function for O(1) access

**Operations:** 
- Insert/Delete/Search: O(1) average, O(n) worst
- Iteration: O(n)

**Collision Handling:**
- Chaining (linked lists at each bucket)
- Open addressing (linear probing, quadratic probing)
- Double hashing

**Use Cases:**
- Caching frequently used data
- Counting frequencies
- Deduplication
- Two-sum problems
- Anagram detection

**Important:**
- Unordered access
- Space vs speed trade-off
- Good hash function crucial for performance

---

### 6. Trees

#### Binary Tree
**Definition:** Max 2 children per node

**Operations:**
- Insert/Delete: O(log n) balanced, O(n) worst
- Search: O(log n) balanced, O(n) worst
- Traversal: O(n)

**Traversal Methods:**
- **Inorder** (Left-Root-Right): sorted in BST
- **Preorder** (Root-Left-Right): copy tree, prefix notation
- **Postorder** (Left-Right-Root): delete tree, postfix notation
- **Level-order** (BFS): use queue

#### Binary Search Tree (BST)
**Property:** Left < Root < Right

**Operations:**
- Search: O(log n) balanced, O(n) worst
- Insert: O(log n) balanced, O(n) worst
- Delete: O(log n) balanced, O(n) worst

**Balance Importance:**
- Unbalanced tree degrades to linked list performance
- AVL, Red-Black trees maintain balance

#### Balanced Trees
**AVL Tree:**
- Height difference ≤ 1
- Rebalance on insert/delete via rotations
- Guaranteed O(log n)

**Red-Black Tree:**
- Color-based balancing
- Less strict than AVL
- Used in Java TreeMap, C++ map

**B-Tree:**
- Multiple children per node
- Used in databases, file systems
- Optimal for disk I/O

#### Heap (Priority Queue)
**Definition:** Complete binary tree with heap property

**Types:**
- **Max Heap:** Parent ≥ Children
- **Min Heap:** Parent ≤ Children

**Operations:**
- Insert: O(log n)
- Delete min/max: O(log n)
- Build heap: O(n)
- Peek min/max: O(1)

**Use Cases:**
- Priority queues
- Heap sort
- Median finding
- K largest/smallest elements

**Implementation:** Array-based
- Parent index: (i-1)/2
- Left child: 2i+1
- Right child: 2i+2

---

### 7. Graphs

**Representation:**
- Adjacency Matrix: O(V²) space, O(1) edge lookup
- Adjacency List: O(V+E) space, O(degree) edge lookup

**Terminology:**
- Vertex (Node), Edge
- Directed vs Undirected
- Weighted vs Unweighted
- Cyclic vs Acyclic
- Connected vs Disconnected
- Dense vs Sparse

**Special Cases:**
- DAG (Directed Acyclic Graph)
- Bipartite graph
- Complete graph
- Tree (connected acyclic graph)

**Traversals:**
- DFS (Depth-First Search): stack-based, recursive
- BFS (Breadth-First Search): queue-based, level-by-level

**Common Algorithms:**
- Shortest path: Dijkstra (O((V+E)log V)), Bellman-Ford (O(VE))
- Minimum spanning tree: Kruskal, Prim
- Topological sort: DFS or Kahn's algorithm
- Cycle detection: DFS, Union-Find

**Use Cases:**
- Social networks (connections)
- Maps (shortest route)
- Dependency resolution
- Bipartite matching

---

### 8. Tries (Prefix Trees)

**Definition:** Tree for efficient prefix searching

**Operations:**
- Insert: O(m) where m = word length
- Search: O(m)
- StartsWith: O(m)
- Space: O(ALPHABET_SIZE * N)

**Use Cases:**
- Autocomplete
- Spell checker
- IP routing
- Word search in grid

**Optimization:**
- TrieNode has children map/array
- End-of-word flag to mark valid words

---

### 9. Union-Find (Disjoint Set Union)

**Purpose:** Track connected components, detect cycles in undirected graphs

**Operations:**
- Find (with path compression): O(α(n)) ≈ O(1)
- Union (with union by rank): O(α(n)) ≈ O(1)

**Key Concepts:**
- Path compression: Make nodes point directly to root
- Union by rank: Attach smaller tree under larger

**Use Cases:**
- Kruskal's algorithm (MST)
- Cycle detection in undirected graphs
- Connected components
- Percolation problems

---

## Algorithms

### 1. Sorting

**Comparison-based Sorting:**

| Algorithm | Best | Average | Worst | Space | Stable | Notes |
|-----------|------|---------|-------|-------|--------|-------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Rarely used, simple |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) | No | Minimal swaps |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) | Yes | Good for small n |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) | Yes | Divide & conquer |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) | No | In-place, cache friendly |
| Heap Sort | O(n log n) | O(n log n) | O(n log n) | O(1) | No | In-place |

**Non-comparison Sorting:**
- **Counting Sort:** O(n+k) space, k = max value, stable
- **Radix Sort:** O(d(n+k)) time, d = digits, stable
- **Bucket Sort:** O(n+k) average, good for uniform distribution

**When to Use:**
- Small dataset: Insertion sort
- General purpose: Merge sort (stable), Quick sort (fast)
- Partially sorted: Insertion sort or Quick sort with random pivot
- Need stability: Merge sort, Bubble sort, Insertion sort

---

### 2. Searching

**Linear Search:** O(n)
- Unordered data
- Small datasets

**Binary Search:** O(log n)
- **Requirement:** Data must be sorted
- **Template:** Left, mid, right pointers
- **Variations:**
  - Find exact value
  - Find first/last occurrence
  - Find closest value
  - Find peak (bitonic array)

**Binary Search Template:**
```
left = 0, right = n - 1
while left <= right:
    mid = left + (right - left) / 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: left = mid + 1
    else: right = mid - 1
return -1
```

---

### 3. Divide and Conquer

**Pattern:**
1. Divide problem into subproblems
2. Conquer recursively
3. Combine results

**Examples:**
- Merge sort
- Quick sort
- Binary search
- Closest pair problem
- Strassen matrix multiplication

---

### 4. Dynamic Programming

**Key Conditions:**
- Overlapping subproblems
- Optimal substructure

**Approaches:**

**Top-Down (Memoization):**
- Recursive with caching
- Store results of subproblems
- Avoids recomputation

**Bottom-Up (Tabulation):**
- Build solution iteratively
- Fill table from base cases
- Better space optimization possible

**Common Patterns:**

**1. 0/1 Knapsack:**
```
dp[i][w] = max profit with first i items and capacity w
dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight[i]] + value[i])
```

**2. Longest Common Subsequence:**
```
dp[i][j] = LCS of first i chars of s1 and first j of s2
if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

**3. Coin Change:**
```
dp[i] = minimum coins to make amount i
dp[i] = min(dp[i - coin] + 1) for all valid coins
```

**4. Edit Distance:**
```
dp[i][j] = min operations to transform s1[0:i] to s2[0:j]
if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
```

**5. House Robber:**
```
dp[i] = max money robbing up to house i
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

**Space Optimization:**
- Use rolling array when only previous state needed
- 2D to 1D conversion possible

---

### 5. Greedy Algorithms

**Pattern:**
- Make locally optimal choice at each step
- Hope to reach global optimum

**Use When:**
- Problem has greedy choice property
- Optimal substructure exists

**Examples:**
- Activity selection (most compatible activities)
- Huffman coding (optimal prefix-free code)
- Kruskal's algorithm (MST)
- Dijkstra's algorithm (shortest path)
- Fractional knapsack
- Coin change (when coin denominations cooperate)

**Warning:** Not all problems have greedy solutions; verify correctness

---

### 6. Graph Algorithms

**DFS (Depth-First Search):**
```
Time: O(V+E)
Space: O(V) - recursion stack
Uses: Cycle detection, topological sort, connected components
```

**BFS (Breadth-First Search):**
```
Time: O(V+E)
Space: O(V) - queue
Uses: Shortest path (unweighted), level-order traversal
```

**Topological Sort:**
- DAG only
- DFS-based or Kahn's algorithm (BFS)
- Applications: Task scheduling, dependency resolution

**Shortest Path:**
- **Unweighted:** BFS O(V+E)
- **Weighted, no negatives:** Dijkstra O((V+E)log V) with min-heap
- **Weighted, with negatives:** Bellman-Ford O(VE)
- **All pairs:** Floyd-Warshall O(V³)

**Minimum Spanning Tree:**
- **Kruskal:** Sort edges, Union-Find, O(E log E)
- **Prim:** Greedy vertex selection, O((V+E)log V)

**Bipartite Check:**
```
Color vertices with 2 colors
If adjacent vertices have same color → not bipartite
Use BFS or DFS with coloring
```

**Cycle Detection:**
- **Undirected:** DFS (back edge) or Union-Find
- **Directed:** DFS (back edge to ancestor)

---

### 7. String Algorithms

**Pattern Matching:**
- **Naive:** O(nm)
- **KMP (Knuth-Morris-Pratt):** O(n+m), build failure function
- **Boyer-Moore:** Fast in practice, O(n/m) best case
- **Rabin-Karp:** Hash-based, good for multiple patterns

**Longest Common Substring:** O(nm) DP

**Longest Palindromic Substring:**
- Expand around center: O(n²)
- DP: O(n²) time, O(n²) space
- Manacher's algorithm: O(n)

**Anagram Detection:**
- Sort both strings: O(n log n)
- Count frequency: O(n)

**Valid Parentheses:**
- Stack-based: O(n) time, O(n) space
- Counter for simple cases

---

## Problem-Solving Patterns

### Two Pointers
**When:** Sorted array, linked list, or need to find pairs

**Examples:**
- Two sum (sorted)
- Valid palindrome
- Container with most water
- Merge sorted arrays
- Remove duplicates

### Sliding Window
**When:** Contiguous subarray/substring with condition

**Pattern:**
```
expand window until condition met
contract when needed to maintain optimal
update result
```

**Examples:**
- Longest substring without repeating characters
- Minimum window substring
- Longest substring of all distinct characters
- Max sum subarray of size k

### Fast & Slow Pointers
**When:** Cycle detection, middle finding

**Examples:**
- Linked list cycle
- Find middle of linked list
- Happy number
- Palindrome linked list

### Merge Intervals
**When:** Overlapping intervals

**Pattern:**
```
sort by start
merge overlapping
```

**Examples:**
- Merge intervals
- Insert interval
- Meeting rooms

### Backtracking
**When:** Generate all combinations/permutations, constraint satisfaction

**Pattern:**
```
if goal reached: record solution
for each choice:
    make choice
    backtrack
    undo choice
```

**Examples:**
- N-Queens
- Sudoku solver
- Word search in grid
- Combination sum
- Permutations/Combinations

### BFS Pattern
**When:** Level-order, shortest path unweighted

**Pattern:**
```
queue = [start]
while queue:
    node = queue.pop(0)
    for neighbor in node.neighbors:
        if not visited:
            queue.append(neighbor)
```

### DFS Pattern
**When:** Traversal, topological sort, cycle detection

**Pattern:**
```
dfs(node):
    if base case: return
    for neighbor in node.neighbors:
        if not visited:
            dfs(neighbor)
```

---

## Common Interview Questions

### Easy Level

1. **Array/String:**
   - Two Sum
   - Valid Parentheses
   - Reverse String/Integer
   - Palindrome
   - Contains Duplicate

2. **Linked List:**
   - Reverse Linked List
   - Merge Two Sorted Lists
   - Linked List Cycle
   - Remove Duplicates

3. **Tree:**
   - Inorder/Preorder/Postorder Traversal
   - Level Order Traversal
   - Maximum Depth
   - Symmetric Tree

### Medium Level

1. **Array/String:**
   - Container with Most Water
   - Longest Substring Without Repeating Characters
   - 3Sum
   - Merge Intervals
   - Rotate Matrix

2. **Linked List:**
   - Add Two Numbers
   - Reorder List
   - Partition List
   - Copy List with Random Pointer

3. **Tree:**
   - Lowest Common Ancestor
   - Serialize/Deserialize
   - Path Sum
   - Vertical Order Traversal

4. **Graph:**
   - Number of Islands
   - Clone Graph
   - Course Schedule (Topological Sort)
   - Word Ladder

5. **Dynamic Programming:**
   - Longest Increasing Subsequence
   - Coin Change
   - House Robber
   - Jump Game

### Hard Level

1. **Array/String:**
   - Median of Two Sorted Arrays
   - Longest Palindromic Substring
   - Wildcard Matching
   - Trapping Rain Water

2. **Tree:**
   - Binary Tree Maximum Path Sum
   - Serialize/Deserialize BST
   - Largest BST in Binary Tree

3. **Graph:**
   - Word Ladder II
   - Alien Dictionary
   - Network Delay Time
   - Minimum Cost to Connect Cities

4. **Dynamic Programming:**
   - Regular Expression Matching
   - Distinct Subsequences
   - Russian Doll Envelopes

---

## Monotonic Stack / Monotonic Deque Pattern

**Core idea:** Maintain a stack where elements are in monotonically increasing or decreasing order. Each element is pushed and popped at most once → **O(n) total**.

### Next Greater Element Template
```python
def next_greater_element(nums):
    result = [-1] * len(nums)
    stack = []  # stores indices
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)
    return result
```

### Monotonic Deque — Sliding Window Maximum
```python
from collections import deque
def max_sliding_window(nums, k):
    dq = deque()  # stores indices, front = max
    result = []
    for i, num in enumerate(nums):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

### Canonical Problems
- **Next Greater Element** (I, II, circular variant)
- **Largest Rectangle in Histogram** — monotonic increasing stack of heights
- **Trapping Rain Water** — monotonic decreasing stack or two-pointer
- **Daily Temperatures** — next warmer day
- **Sliding Window Maximum** — monotonic deque

---

## Bit Manipulation

### Key Operations
| Operation | Syntax | Use |
|-----------|--------|-----|
| AND | `a & b` | Mask bits, check if bit set |
| OR | `a \| b` | Set bits |
| XOR | `a ^ b` | Toggle bits, find unique element |
| NOT | `~a` | Flip all bits |
| Left shift | `a << n` | Multiply by 2ⁿ |
| Right shift | `a >> n` | Divide by 2ⁿ |

### Essential Tricks
- **`n & (n-1)`** — clear lowest set bit. If result is 0 → n is a power of 2.
- **`n & (-n)`** — isolate lowest set bit (used in Fenwick trees).
- **`a ^ a = 0`** — XOR identity. Find single number in array where all others appear twice.
- **Brian Kernighan's bit count:** count set bits in O(set bits) by repeatedly clearing lowest set bit.

```python
def count_bits(n):
    count = 0
    while n:
        n &= n - 1  # clear lowest set bit
        count += 1
    return count
```

### Bitmask DP
Represent subsets as integers. For a set of n elements, iterate over all 2ⁿ subsets.
```python
# Enumerate all subsets of a bitmask
mask = (1 << n) - 1  # all n bits set
sub = mask
while sub:
    # process subset 'sub'
    sub = (sub - 1) & mask
```

### Common Problems
- Single Number (XOR all elements)
- Counting Bits (DP: `dp[i] = dp[i >> 1] + (i & 1)`)
- Subsets (bitmask enumeration)
- Hamming Distance (`bin(x ^ y).count('1')`)
- TSP with bitmask DP: `dp[mask][i]` = min cost visiting cities in mask, ending at i

---

## Prefix Sum and Difference Arrays

### 1D Prefix Sum
```python
# Build: O(n), Query any range sum: O(1)
prefix = [0] * (len(arr) + 1)
for i in range(len(arr)):
    prefix[i+1] = prefix[i] + arr[i]
# Sum of arr[l..r] = prefix[r+1] - prefix[l]
```

### Subarray Sum Equals K (Hashmap Technique)
```python
def subarray_sum(nums, k):
    count, curr_sum = 0, 0
    prefix_counts = {0: 1}
    for num in nums:
        curr_sum += num
        count += prefix_counts.get(curr_sum - k, 0)
        prefix_counts[curr_sum] = prefix_counts.get(curr_sum, 0) + 1
    return count
```

### 2D Prefix Sum
```python
# Build 2D prefix sum matrix for O(1) rectangle queries
# sum of rect (r1,c1)→(r2,c2) = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]
```

### Difference Array
- **O(1) range updates**, O(n) reconstruction.
- To add val to range [l, r]: `diff[l] += val; diff[r+1] -= val`
- Reconstruct with prefix sum of diff array.
- Use case: multiple range update queries then single read pass.

---

## Advanced DP Patterns

### Interval DP
Solve subproblems defined by intervals [i, j]. Iterate by length, then start index, try all split points.
```python
# Matrix Chain Multiplication / Burst Balloons pattern
for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(i, j):  # split point
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i, k, j))
```

### Bitmask DP
Represent visited/chosen subset as bitmask integer.
```python
# TSP: dp[mask][i] = min cost to visit cities in 'mask' ending at city i
for mask in range(1 << n):
    for i in range(n):
        if not (mask & (1 << i)): continue
        for j in range(n):
            if mask & (1 << j): continue
            new_mask = mask | (1 << j)
            dp[new_mask][j] = min(dp[new_mask][j], dp[mask][i] + dist[i][j])
```

### Tree DP
DFS on tree, compute DP values bottom-up.
```python
# House Robber III: for each node, track (rob_this, skip_this)
def dfs(node):
    if not node: return (0, 0)
    left = dfs(node.left)
    right = dfs(node.right)
    rob = node.val + left[1] + right[1]
    skip = max(left) + max(right)
    return (rob, skip)
```

### Digit DP
Count numbers up to N with certain digit properties. Track (position, tight_constraint, accumulated_state).
- Used for: count numbers with digit sum = K, numbers without consecutive same digits, etc.

---

## Amortized Analysis

**Concept:** Some operations are expensive but happen infrequently enough that the average cost per operation is low.

### Examples
| Operation | Worst case | Amortized | Why |
|-----------|-----------|-----------|-----|
| Dynamic array append | O(n) when doubling | **O(1)** | Doubling costs O(n) but happens every n insertions |
| Stack multipop | O(n) per multipop | **O(1)** per operation | Each element pushed/popped at most once total |
| Union-Find with path compression | O(log n) | **O(α(n)) ≈ O(1)** | Inverse Ackermann via path compression + union by rank |

### Methods
- **Aggregate analysis:** total cost of n operations / n
- **Accounting method:** assign amortized cost per operation, "bank" surplus for expensive ops
- **Potential method:** define potential function Φ, amortized cost = actual cost + ΔΦ

### Interview tip
When asked "isn't ArrayList.add() O(n)?", explain: "Worst case is O(n) when the array doubles, but this happens once every n insertions. By aggregate analysis, n insertions cost O(n) total, so amortized cost per insertion is O(1)."

---

## Two-Heap Pattern

Use a **max-heap for the lower half** and **min-heap for the upper half** to maintain the running median in O(log n) per insertion.

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (negate values)
        self.hi = []  # min-heap

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

### Applications
- **Find Median from Data Stream** (LeetCode 295)
- **Sliding Window Median** (LeetCode 480)
- **IPO** — maximize capital with k investments

---

## ML/AI Interview Bridge — DSA Meets ML

| DSA Concept | ML Application |
|---|---|
| Hash tables + random projection | Locality-Sensitive Hashing (LSH) for approximate nearest neighbor |
| BFS with pruning | Beam search in sequence generation (LLMs, NMT) |
| Priority queues (heaps) | Top-K sampling, best-first search in inference |
| Tries | Tokenizer vocabulary lookup (BPE merge rules) |
| Graphs + topological sort | Computation graphs in autograd (PyTorch backward pass) |
| Dynamic programming | Viterbi algorithm (HMMs, CRF decoding), CTC loss |
| Divide and conquer | MapReduce paradigm, distributed data sharding |
| Segment trees / Fenwick trees | Efficient range queries in feature stores |
| Union-Find | Connected components in entity resolution / deduplication |
| Monotonic stack | Efficient attention masking, sequence processing |
| Prefix sums | Cumulative distribution functions, integral images |

---

## Interview Tips

1. **Clarify Requirements:**
   - Ask about input constraints
   - Edge cases (empty, single element, duplicates)
   - Expected output format

2. **Think Out Loud:**
   - Explain your approach before coding
   - Discuss trade-offs
   - Consider time/space complexity

3. **Start Simple:**
   - Brute force first, then optimize
   - Handle easy cases first
   - Incrementally improve

4. **Test Thoroughly:**
   - Happy path
   - Edge cases (empty, single, duplicate)
   - Large inputs
   - Negative numbers/special values

5. **Optimization Checklist:**
   - Can I use different data structure?
   - Can I reduce time complexity?
   - Can I reduce space complexity?
   - Are there patterns I can use?

6. **Code Quality:**
   - Clear variable names
   - Consistent formatting
   - Handle edge cases
   - Add comments for complex logic

---

## Quick Reference: When to Use What

| Problem Type | Data Structure | Algorithm |
|--------------|-----------------|-----------|
| Ordering | Array | Sort (Merge/Quick) |
| Search | Sorted Array | Binary Search |
| Grouping | Hash Map | Iteration |
| Frequency | Hash Map/Counter | Count |
| Top K | Heap | Heap operations |
| Running Median | Two Heaps (max+min) | Push/pop/balance |
| Next Greater/Smaller | Monotonic Stack | Single pass O(n) |
| Sliding Window Max/Min | Monotonic Deque | Deque maintenance |
| Range Sum Query | Prefix Sum Array | Precompute + O(1) lookup |
| Range Updates | Difference Array | Batch update + reconstruct |
| Single Unique Element | - | XOR all elements |
| Subset Enumeration | Bitmask | Bitmask DP |
| Relationships | Graph | DFS/BFS |
| Dependencies | Graph | Topological Sort |
| Shortest Path | Graph | Dijkstra/BFS |
| Substring | String/Trie | KMP/DP |
| Subsequence | Array | DP |
| Interval Problems | Array | Interval DP |
| Tree Optimization | Tree | Tree DP (post-order DFS) |
| Combination/Permutation | - | Backtracking |
| Overlapping Subproblems | - | DP |

---

## Resources for Practice

- **LeetCode:** Categorized by difficulty and topic
- **HackerRank:** Interactive coding challenges
- **InterviewBit:** Topic-wise learning paths
- **GeeksforGeeks:** Algorithm explanations
- **GitHub:** Solution repositories

---

**Last Updated:** July 2026

Remember: Understanding concepts > Memorizing solutions. Code regularly, analyze complexity, and practice explaining your approach!
