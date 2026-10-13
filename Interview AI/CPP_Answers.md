# C++ Practice — Answers

Reference answers for `CPP_Questions.md`. Read only after attempting each question.

---

## 1. Basics

**A1. Fundamental data types**

| Type | Size (typical) | Example |
|------|---------------|---------|
| `int` | 4 bytes | `int x = 5;` |
| `float` | 4 bytes | `float f = 3.14f;` |
| `double` | 8 bytes | `double d = 3.14;` |
| `char` | 1 byte | `char c = 'A';` |
| `bool` | 1 byte | `bool b = true;` |
| `long long` | 8 bytes | `long long n = 1e18;` |
| `void` | — | functions returning nothing |

---

**A2. int vs unsigned int vs long vs long long**

- `int` — 32-bit, range ~−2.1B to 2.1B
- `unsigned int` — 32-bit, no negatives, range 0 to ~4.2B
- `long` — platform-dependent (32-bit Windows, 64-bit Linux/Mac)
- `long long` — always 64-bit, range ~−9.2×10¹⁸ to 9.2×10¹⁸

Use `long long` when results could overflow `int`.

---

**A3. auto**

`auto` lets the compiler deduce the type from the initializer.

```cpp
auto x = 5;           // int
auto f = 3.14;        // double
auto v = std::vector<int>{1, 2, 3};
```

Use when: type is obvious, or the type is complex (iterators, lambdas).
Avoid when: type is non-obvious and hurts readability.

---

**A4. =, ==, ===**

- `=` — assignment: `x = 5;`
- `==` — equality check: `if (x == 5)`
- `===` — **does not exist in C++** (JavaScript only)

Common bug: `if (x = 5)` assigns 5 and evaluates as true.

---

**A5. const**

Marks a variable as non-modifiable after initialization.

```cpp
const int MAX = 100;
MAX = 200;   // compile error
```

Why: prevents accidents, communicates intent, enables optimizations, required for const references.

---

## 2. Control Flow

**A6. Loop types**

```cpp
for (int i = 0; i < n; i++) { }        // known iteration count
while (condition) { }                   // unknown count, check first
do { } while (condition);              // must run at least once
for (int x : vec) { }                  // iterate over container (C++11)
```

---

**A7. break vs continue**

- `break` — exits the loop entirely
- `continue` — skips current iteration, moves to next

```cpp
for (int i = 0; i < 5; i++) {
    if (i == 2) continue;   // skip 2
    if (i == 4) break;      // stop at 4
    std::cout << i;         // prints: 0 1 3
}
```

---

**A8. switch and fall-through**

```cpp
switch (x) {
    case 1:
        std::cout << "one";
        break;
    case 2:
        std::cout << "two";
        break;
    default:
        std::cout << "other";
}
```

Fall-through: omitting `break` causes execution to continue into the next case. Sometimes intentional (shared logic), usually a bug.

---

## 3. Functions

**A9. Function overloading**

Same name, different parameter types or count — compiler picks the right one.

```cpp
int add(int a, int b)          { return a + b; }
double add(double a, double b) { return a + b; }
```

**Cannot** use return type alone to differentiate overloads.

---

**A10. Pass-by-value vs reference vs pointer**

```cpp
void byValue(int x)  { x = 10; }   // copy, caller unchanged
void byRef(int& x)   { x = 10; }   // alias, caller IS changed
void byPtr(int* x)   { *x = 10; }  // pointer, caller IS changed
```

- By value: safe, copies data — expensive for large objects
- By reference: no copy, modifies original
- By const reference: no copy, read-only — best for large objects you don't modify
- By pointer: when nullability or pointer reseating is needed

---

**A11. Default arguments**

Pre-assigned values used when caller doesn't provide them.

```cpp
void greet(std::string name, std::string msg = "Hello") {
    std::cout << msg << ", " << name;
}
greet("Alice");         // "Hello, Alice"
greet("Alice", "Hi");  // "Hi, Alice"
```

Rule: default arguments must be at the **end** of the parameter list.

---

**A12. Recursion and base case**

A function calling itself. Must have:
1. **Base case** — stops recursion
2. **Recursive case** — calls itself with simpler input

```cpp
int factorial(int n) {
    if (n <= 1) return 1;        // base case
    return n * factorial(n - 1); // recursive case
}
```

Without base case: infinite recursion → stack overflow.

---

## 4. Pointers & References

**A13. Pointers**

```cpp
int x = 5;
int* p = &x;    // p holds address of x
*p = 10;        // dereference: change value at address
std::cout << *p;   // 10
std::cout << p;    // memory address (e.g. 0x7ffd...)
```

- `&x` — address-of
- `*p` — dereference

---

**A14. Null pointer**

A pointer that doesn't point to anything valid.

```cpp
int* p = nullptr;   // modern (preferred)
int* p = NULL;      // old C style (avoid)

if (p != nullptr) {
    *p = 5;   // always check before dereferencing
}
```

Dereferencing a null pointer is **undefined behavior** — usually crashes.

---

**A15. Pointer vs reference**

| Feature | Pointer | Reference |
|---------|---------|-----------|
| Access syntax | `*p` | direct (`ref`) |
| Can be null | yes | no |
| Can be reseated | yes | no |
| Must be initialized | no | yes |
| Supports arithmetic | yes | no |

Use references when it's always valid. Use pointers for nullability or reseating.

---

**A16. const pointer positions**

Read right-to-left from the variable name:

- `const int* p` — pointer to const int: can repoint `p`, **cannot** change `*p`
- `int* const p` — const pointer to int: **cannot** repoint `p`, can change `*p`
- `const int* const p` — neither allowed

```cpp
int a = 1, b = 2;
const int* p = &a;
p = &b;    // ok
*p = 5;    // error

int* const q = &a;
q = &b;    // error
*q = 5;    // ok
```

---

**A17. Pointer arithmetic**

Math on addresses, scaled by type size.

```cpp
int arr[] = {10, 20, 30};
int* p = arr;

*(p + 1)   // 20 — moved 4 bytes (sizeof int)
*(p + 2)   // 30
```

`p + n` moves `n * sizeof(type)` bytes. `arr[i]` is exactly `*(arr + i)`.

---

**A18. Double pointer**

A pointer to a pointer. Used to modify a pointer from inside a function.

```cpp
void allocate(int** pp) {
    *pp = new int(100);   // changes the caller's pointer
}

int* p = nullptr;
allocate(&p);             // p now points to heap int = 100
delete p;
```

Without `**`, the function only modifies a local copy of the pointer.

---

## 5. Arrays & Strings

**A19. C-style arrays**

```cpp
int arr[5] = {1, 2, 3, 4, 5};
int arr2[5] = {1, 2};      // {1, 2, 0, 0, 0}
int arr3[] = {1, 2, 3};    // size deduced as 3
```

Arrays do **not** carry their size — you must pass size separately, or use `std::array`/`std::vector`.

---

**A20. C-string vs std::string**

```cpp
// C-string
char s[] = "hello";
strlen(s);             // manual length
strcpy, strcmp         // manual operations, error-prone

// std::string
std::string s = "hello";
s.length();            // 5
s + " world";          // concatenation
s == "hello";          // comparison
```

Prefer `std::string`. Use C-strings only when interfacing with C APIs.

---

**A21. std::vector vs array**

```cpp
std::vector<int> v = {1, 2, 3};
v.push_back(4);
v.size();           // 4
v.pop_back();
v.erase(v.begin() + 1);
```

| Feature | Array | Vector |
|---------|-------|--------|
| Size | fixed | dynamic |
| Memory | stack | heap |
| Tracks its size | no | yes |
| Bounds checking | no | `.at()` does |

---

## 6. Object-Oriented Programming

**A22. class vs struct**

Both group data and functions. Only difference: `class` defaults to `private`, `struct` defaults to `public`.

Convention: `struct` for plain data, `class` for objects with behavior.

---

**A23. Constructor and destructor**

- **Constructor** — called on object creation, initializes the object
- **Destructor** — called on object destruction, cleans up resources

```cpp
class Foo {
public:
    Foo()      { std::cout << "born"; }   // constructor
    Foo(int x) { val = x; }              // parameterized
    ~Foo()     { std::cout << "dead"; }  // destructor
private:
    int val;
};

{ Foo f(42); }   // constructor then destructor when leaving scope
```

---

**A24. Access specifiers**

- `public` — accessible from anywhere
- `private` — only within the class itself
- `protected` — within the class and derived classes

```cpp
class Base {
public:    int a;   // anyone
protected: int b;   // Base + Derived
private:   int c;   // only Base
};
```

---

**A25. Inheritance**

A derived class inherits members and methods from a base class.

```cpp
class Animal {
public:
    void eat() { std::cout << "eating"; }
};

class Dog : public Animal {
public:
    void bark() { std::cout << "woof"; }
};

Dog d;
d.eat();    // inherited from Animal
d.bark();   // Dog's own method
```

---

**A26. Polymorphism and virtual**

Polymorphism = calling the correct method based on **runtime type**, not declared type.

```cpp
class Shape {
public:
    virtual void draw() { std::cout << "shape"; }
};

class Circle : public Shape {
public:
    void draw() override { std::cout << "circle"; }
};

Shape* s = new Circle();
s->draw();   // "circle" — correct runtime type used
```

Without `virtual`: always calls base version ("shape") regardless of actual type.

---

**A27. Pure virtual and abstract class**

Pure virtual = `= 0`, no implementation, must be overridden.

```cpp
class Shape {
public:
    virtual void draw() = 0;   // pure virtual
};

class Circle : public Shape {
public:
    void draw() override { std::cout << "circle"; }
};

Shape s;    // error — abstract class can't be instantiated
Circle c;   // ok
```

Any class with at least one pure virtual function is abstract.

---

**A28. Rule of Three / Rule of Five**

If you define any one of these, define all:

**Rule of Three:**
1. Destructor
2. Copy constructor
3. Copy assignment operator

**Rule of Five (C++11):** adds:
4. Move constructor
5. Move assignment operator

Applies when your class manually manages a resource (heap memory, file, socket).

---

## 7. Memory Management

**A29. Stack vs heap**

| Feature | Stack | Heap |
|---------|-------|------|
| Allocation | automatic | manual (`new`) |
| Deallocation | automatic | manual (`delete`) |
| Size | small (~1-8MB) | large (GBs) |
| Speed | fast | slower |

---

**A30. Memory leak**

Heap memory allocated but never freed — pointer is lost but memory stays occupied.

```cpp
void leak() {
    int* p = new int(5);
    return;   // p gone, heap int never deleted
}

// fix with delete:
void fixed() {
    int* p = new int(5);
    delete p;
}

// or better, use smart pointer:
void better() {
    auto p = std::make_unique<int>(5);
    // auto-freed when p goes out of scope
}
```

---

**A31. new/delete vs new[]/delete[]**

```cpp
int* p = new int(42);      // single
delete p;

int* arr = new int[10];    // array
delete[] arr;              // must match with delete[]
```

Mismatching (using `delete` on array) is **undefined behavior**.
Prefer `std::make_unique` over raw `new` in modern C++.

---

**A32. Smart pointer types**

| Type | Ownership | Use case |
|------|-----------|----------|
| `unique_ptr` | exclusive, move-only | default choice |
| `shared_ptr` | shared, ref-counted | multiple owners |
| `weak_ptr` | none | break shared_ptr cycles, caches |

```cpp
auto u = std::make_unique<int>(42);
auto s1 = std::make_shared<int>(42);
auto s2 = s1;                          // ref count = 2
std::weak_ptr<int> w = s1;
```

---

## 8. Templates

**A33. Function template**

```cpp
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

max(3, 5);      // int
max(3.0, 5.0);  // double
```

Compiler generates a concrete version for each type — called **template instantiation**.

---

**A34. Class template**

```cpp
template <typename T>
class Stack {
    std::vector<T> data;
public:
    void push(T val) { data.push_back(val); }
    T pop() {
        T v = data.back();
        data.pop_back();
        return v;
    }
    bool empty() { return data.empty(); }
};

Stack<int> s;
s.push(1);
s.pop();   // 1
```

---

## 9. STL

**A35. STL containers**

| Container | Description |
|-----------|-------------|
| `vector` | dynamic array, O(1) random access |
| `deque` | double-ended queue |
| `list` | doubly linked list, O(1) insert anywhere |
| `set` | sorted unique elements, O(log n) |
| `map` | sorted key-value pairs, O(log n) |
| `unordered_set` | hash set, O(1) avg |
| `unordered_map` | hash map, O(1) avg |
| `stack` | LIFO adapter |
| `queue` | FIFO adapter |
| `priority_queue` | max-heap |

---

**A36. map vs unordered_map**

```cpp
std::map<std::string, int> m;           // sorted, O(log n)
std::unordered_map<std::string, int> u; // hash, O(1) avg
```

| Feature | `map` | `unordered_map` |
|---------|-------|-----------------|
| Structure | red-black tree | hash table |
| Keys | sorted | unsorted |
| Lookup | O(log n) | O(1) avg |
| Worst case | O(log n) | O(n) |

Pick `map` when you need sorted keys. Pick `unordered_map` for faster lookups.

---

**A37. STL algorithms**

```cpp
#include <algorithm>

std::sort(v.begin(), v.end());
std::reverse(v.begin(), v.end());
std::find(v.begin(), v.end(), 4);       // iterator to element
std::count(v.begin(), v.end(), 1);      // occurrences
std::min_element(v.begin(), v.end());
std::for_each(v.begin(), v.end(), fn);
```

---

## 10. Modern C++

**A38. Lambda expressions**

```cpp
auto add = [](int a, int b) { return a + b; };

int x = 10;
auto addX  = [x](int n)  { return n + x; };   // capture by value
auto addX2 = [&x](int n) { x += n; };          // capture by reference
```

Capture list: `[]` none, `[=]` all by value, `[&]` all by ref, `[x, &y]` mixed.

---

**A39. nullptr vs NULL**

```cpp
void foo(int x)  { }
void foo(int* p) { }

foo(NULL);      // ambiguous — NULL is 0, could match int version
foo(nullptr);   // unambiguous — always matches pointer version
```

Always prefer `nullptr` in modern C++.

---

**A40. Move semantics and std::move**

Move transfers resources instead of copying — O(1) vs O(n) for large objects.

```cpp
std::vector<int> a = {1, 2, 3, 4, 5};
std::vector<int> b = std::move(a);   // a's data stolen by b, no copy
// a is now empty
```

`std::move` casts to an rvalue reference — it doesn't move anything itself, it signals that moving is allowed.

---

**A41. Range-based for**

```cpp
for (int x : v)            { }   // copy each element
for (int& x : v)           { }   // reference, can modify
for (const int& x : v)     { }   // const ref, read-only, no copy
for (const auto& x : v)    { }   // best for complex types
```

---

**A42. constexpr vs const**

- `const` — value fixed at runtime, cannot change after init
- `constexpr` — value computed at **compile time**

```cpp
constexpr int square(int x) { return x * x; }
constexpr int val = square(5);   // 25 at compile time
int arr[square(3)];              // ok — needs compile-time constant
```

---

## 11. Exception Handling

**A43. try/catch**

```cpp
try {
    if (x < 0) throw std::invalid_argument("negative");
} catch (const std::invalid_argument& e) {
    std::cout << e.what();   // "negative"
} catch (const std::exception& e) {
    std::cout << e.what();
} catch (...) {
    // catch anything
}
```

`e.what()` returns the error message string.

---

**A44. noexcept**

Marks a function as guaranteed not to throw.

```cpp
void foo() noexcept { }
```

If it throws anyway: `std::terminate` is called immediately.
Benefits: enables compiler optimizations, required for move constructors in some STL contexts.

---

## 12. Miscellaneous

**A45. ++i vs i++**

- `++i` — pre-increment: increments, returns **new** value
- `i++` — post-increment: returns **old** value, then increments

```cpp
int i = 5;
int a = ++i;   // i=6, a=6
int b = i++;   // b=6, then i=7
```

Prefer `++i` for iterators — `i++` requires creating a temporary copy of the iterator.

---

**A46. Operator overloading**

Define custom behavior for operators on your own types.

```cpp
class Vec2 {
public:
    float x, y;
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }
};

Vec2 a(1, 2), b(3, 4);
Vec2 c = a + b;   // calls operator+, c = (4, 6)
```

---

**A47. static in class vs in function**

**Static class member** — one copy shared across all instances:
```cpp
class Counter {
public:
    static int count;
    Counter() { count++; }
};
int Counter::count = 0;   // define outside class

Counter a, b, c;
Counter::count;   // 3
```

**Static local variable** — initialized once, persists across calls:
```cpp
int nextId() {
    static int id = 0;   // only initialized first time
    return ++id;
}
nextId();   // 1
nextId();   // 2
```

---

**A48. Struct initialization styles**

```cpp
struct Point { int x, y; };

Point p1;            // uninitialized (undefined values)
Point p2 = {1, 2};  // aggregate initialization
Point p3{1, 2};      // uniform initialization (C++11, preferred)
Point p4 = {};       // zero-initialized: {0, 0}
Point p5{};          // same as above
```

Uniform initialization (`{}`) is preferred — works everywhere, prevents narrowing conversions.

---

## Rapid Fire Answers

1. `&` in declaration = reference (alias). `&` in expression = address-of.
2. `*` in declaration = pointer type. `*` in expression = dereference.
3. `bool` is 1 byte (8 bits), but only uses 1 bit logically.
4. No — a reference cannot be null. It must always bind to a valid object.
5. Yes — a pointer can be reseated (reassigned to point elsewhere).
6. `class` defaults to `private`. `struct` defaults to `public`.
7. `override` tells the compiler you're overriding a virtual function — catches typos and missing `virtual` in base.
8. A vtable (virtual dispatch table) is a per-class table of function pointers used to resolve `virtual` calls at runtime.
9. `delete[]` calls the destructor on every element in the array before freeing. `delete` only destructs one element.
10. RAII (Resource Acquisition Is Initialization) — resources are acquired in the constructor and released in the destructor, tying resource lifetimes to object scope.
