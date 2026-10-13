# C++ Practice Questions — Full Intro

Questions are grouped by topic. Try to answer before reading the answer.

---

## 1. Basics

**Q1. What are the fundamental data types in C++?**

<details><summary>Answer</summary>

| Type | Size (typical) | Example |
|------|---------------|---------|
| `int` | 4 bytes | `int x = 5;` |
| `float` | 4 bytes | `float f = 3.14f;` |
| `double` | 8 bytes | `double d = 3.14;` |
| `char` | 1 byte | `char c = 'A';` |
| `bool` | 1 byte | `bool b = true;` |
| `long long` | 8 bytes | `long long n = 1e18;` |
| `void` | — | used for functions returning nothing |

</details>

---

**Q2. What is the difference between `int`, `unsigned int`, `long`, and `long long`?**

<details><summary>Answer</summary>

- `int` — typically 32-bit, range: ~−2.1B to 2.1B
- `unsigned int` — 32-bit, no negatives, range: 0 to ~4.2B
- `long` — platform-dependent (32-bit on Windows, 64-bit on Linux/Mac)
- `long long` — always 64-bit, range: ~−9.2×10¹⁸ to 9.2×10¹⁸

Use `long long` when you might overflow `int` (e.g. competitive programming).

</details>

---

**Q3. What is `auto` and when should you use it?**

<details><summary>Answer</summary>

`auto` lets the compiler deduce the type from the initializer.

```cpp
auto x = 5;           // int
auto f = 3.14;        // double
auto s = "hello";     // const char*
auto v = std::vector<int>{1, 2, 3};  // std::vector<int>
```

Use it when:
- The type is obvious from context (reduces verbosity)
- The type is complex (iterators, lambdas)

Avoid it when:
- The type is not obvious and readability suffers

</details>

---

**Q4. What is the difference between `=`, `==`, and `===` in C++?**

<details><summary>Answer</summary>

- `=` — assignment operator: `x = 5;`
- `==` — equality comparison: `if (x == 5)`
- `===` — **does not exist in C++** (that's JavaScript)

Common bug: `if (x = 5)` assigns 5 to x and evaluates as true. Always use `==` for comparison.

</details>

---

**Q5. What is `const` and why use it?**

<details><summary>Answer</summary>

`const` marks a variable as non-modifiable after initialization.

```cpp
const int MAX = 100;    // MAX can never be changed
MAX = 200;              // compile error
```

Why use it:
- Prevents accidental modification
- Communicates intent to other developers
- Enables compiler optimizations
- Required for passing by const reference

</details>

---

## 2. Control Flow

**Q6. What are the loop types in C++ and when to use each?**

<details><summary>Answer</summary>

```cpp
// for — when you know the number of iterations
for (int i = 0; i < n; i++) { }

// while — when you don't know iterations upfront
while (condition) { }

// do-while — when you need to run at least once
do { } while (condition);

// range-based for — iterate over containers (C++11)
for (int x : vec) { }
for (auto& x : vec) { }   // by reference, avoids copying
```

</details>

---

**Q7. What is the difference between `break` and `continue`?**

<details><summary>Answer</summary>

- `break` — exits the loop entirely
- `continue` — skips the rest of the current iteration and goes to the next

```cpp
for (int i = 0; i < 5; i++) {
    if (i == 2) continue;   // skip 2
    if (i == 4) break;      // stop at 4
    std::cout << i;         // prints: 0 1 3
}
```

</details>

---

**Q8. What is a switch statement? What is fall-through?**

<details><summary>Answer</summary>

```cpp
switch (x) {
    case 1:
        std::cout << "one";
        break;          // without break, falls through to next case
    case 2:
        std::cout << "two";
        break;
    default:
        std::cout << "other";
}
```

Fall-through: if you omit `break`, execution continues into the next case. Sometimes intentional, often a bug.

</details>

---

## 3. Functions

**Q9. What is function overloading?**

<details><summary>Answer</summary>

Multiple functions with the same name but different parameter types or counts. The compiler picks the right one based on arguments.

```cpp
int add(int a, int b)       { return a + b; }
double add(double a, double b) { return a + b; }

add(1, 2);       // calls int version
add(1.0, 2.0);   // calls double version
```

Return type alone cannot differentiate overloaded functions.

</details>

---

**Q10. What is the difference between pass-by-value, pass-by-reference, and pass-by-pointer?**

<details><summary>Answer</summary>

```cpp
void byValue(int x)    { x = 10; }   // copy, caller unchanged
void byRef(int& x)     { x = 10; }   // alias, caller IS changed
void byPtr(int* x)     { *x = 10; }  // pointer, caller IS changed

int a = 5;
byValue(a);   // a still 5
byRef(a);     // a is now 10
byPtr(&a);    // a is now 10
```

- Pass-by-value: safe, but copies data (expensive for large objects)
- Pass-by-reference: no copy, modifies original
- Pass-by-const-reference: no copy, read-only — best for large objects you don't want to modify

```cpp
void print(const std::string& s) { }  // best practice for large objects
```

</details>

---

**Q11. What are default arguments?**

<details><summary>Answer</summary>

Parameters with pre-assigned values used when the caller doesn't provide them.

```cpp
void greet(std::string name, std::string msg = "Hello") {
    std::cout << msg << ", " << name;
}

greet("Alice");           // "Hello, Alice"
greet("Alice", "Hi");     // "Hi, Alice"
```

Rules:
- Default arguments must be at the **end** of the parameter list
- Can't have a non-default argument after a default one

</details>

---

**Q12. What is recursion? What is a base case?**

<details><summary>Answer</summary>

A function that calls itself. Every recursive function needs:
1. **Base case** — condition to stop recursion
2. **Recursive case** — the function calling itself with a simpler input

```cpp
int factorial(int n) {
    if (n <= 1) return 1;       // base case
    return n * factorial(n - 1); // recursive case
}
```

Without a base case: infinite recursion → stack overflow.

</details>

---

## 4. Pointers & References

**Q13. What is a pointer? How do you declare and use one?**

<details><summary>Answer</summary>

A pointer stores the memory address of another variable.

```cpp
int x = 5;
int* p = &x;    // p holds the address of x

*p = 10;        // dereference: change value at address p
std::cout << *p;  // prints 10
std::cout << p;   // prints memory address (e.g. 0x7ffd...)
```

- `&x` — address-of operator
- `*p` — dereference operator (go to address, get value)

</details>

---

**Q14. What is a null pointer? What is `nullptr`?**

<details><summary>Answer</summary>

A pointer that doesn't point to anything valid.

```cpp
int* p = nullptr;   // modern C++ (preferred)
int* p = NULL;      // old C style (avoid)
int* p = 0;         // also valid but less clear

if (p != nullptr) {
    *p = 5;   // safe — always check before dereferencing
}
```

Dereferencing a null pointer is **undefined behavior** (usually crashes).

</details>

---

**Q15. What is the difference between a pointer and a reference?**

<details><summary>Answer</summary>

| Feature | Pointer | Reference |
|---------|---------|-----------|
| Syntax to access | `*p` | `ref` (direct) |
| Can be null | yes | no |
| Can be reseated | yes (`p = &other`) | no |
| Must be initialized | no | yes |
| Arithmetic | yes (`p++`) | no |

Use references when you know it always refers to something valid. Use pointers when nullability or reseating is needed.

</details>

---

**Q16. What are `const int*`, `int* const`, and `const int* const`?**

<details><summary>Answer</summary>

Read right-to-left from the variable name:

- `const int* p` — pointer to const int: can repoint `p`, cannot change `*p`
- `int* const p` — const pointer to int: cannot repoint `p`, can change `*p`
- `const int* const p` — const pointer to const int: neither

```cpp
int a = 1, b = 2;
const int* p = &a;
p = &b;    // ok
*p = 5;    // error

int* const q = &a;
q = &b;    // error
*q = 5;    // ok
```

</details>

---

**Q17. What is pointer arithmetic?**

<details><summary>Answer</summary>

Math on pointer addresses. The compiler scales by the type size.

```cpp
int arr[] = {10, 20, 30, 40};
int* p = arr;

*(p + 1)   // 20 — moves 4 bytes (sizeof int)
*(p + 2)   // 30
p++        // p now points to arr[1]
```

`arr[i]` is exactly equivalent to `*(arr + i)`.

</details>

---

**Q18. What is a double pointer (`**`)?**

<details><summary>Answer</summary>

A pointer to a pointer. Used when you need to modify a pointer from inside a function.

```cpp
void allocate(int** pp) {
    *pp = new int(100);  // changes the caller's pointer
}

int* p = nullptr;
allocate(&p);            // p now points to heap int with value 100
delete p;
```

Also used for 2D arrays and arrays of strings in C-style code.

</details>

---

## 5. Arrays & Strings

**Q19. How are arrays declared and used in C++?**

<details><summary>Answer</summary>

```cpp
int arr[5] = {1, 2, 3, 4, 5};   // fixed size, stack allocated
arr[0]   // 1
arr[4]   // 5

// partial initialization — rest zero-filled
int arr2[5] = {1, 2};    // {1, 2, 0, 0, 0}

// size deduced
int arr3[] = {1, 2, 3};  // size 3
```

Arrays don't carry their size — pass size separately or use `std::array`/`std::vector`.

</details>

---

**Q20. What is the difference between a C-string and `std::string`?**

<details><summary>Answer</summary>

**C-string** (`char*` or `char[]`):
```cpp
char s[] = "hello";    // null-terminated array of chars
strlen(s)              // 5, doesn't count '\0'
strcpy, strcmp, strcat // manual operations, error-prone
```

**`std::string`**:
```cpp
std::string s = "hello";
s.length()    // 5
s + " world"  // concatenation
s == "hello"  // comparison
s[0]          // 'h'
```

Prefer `std::string` in modern C++. C-strings only when interfacing with C APIs.

</details>

---

**Q21. What is `std::vector` and how does it differ from an array?**

<details><summary>Answer</summary>

`std::vector` is a dynamic array — it can resize at runtime.

```cpp
#include <vector>

std::vector<int> v = {1, 2, 3};
v.push_back(4);         // {1, 2, 3, 4}
v.size()                // 4
v[0]                    // 1
v.pop_back();           // removes last element
v.erase(v.begin() + 1); // removes index 1
```

| Feature | Array | Vector |
|---------|-------|--------|
| Size | fixed | dynamic |
| Memory | stack | heap |
| Size tracking | no | yes (`.size()`) |
| Bounds checking | no | `.at()` does |

</details>

---

## 6. Object-Oriented Programming

**Q22. What is a class? How does it differ from a struct?**

<details><summary>Answer</summary>

Both are user-defined types that group data and functions.

```cpp
class Player {
private:              // default access
    int health;
public:
    void damage(int n) { health -= n; }
};

struct Point {
    int x, y;         // public by default
};
```

The only difference: `class` defaults to `private`, `struct` defaults to `public`. Convention: use `struct` for plain data, `class` for objects with behavior.

</details>

---

**Q23. What is a constructor? What is a destructor?**

<details><summary>Answer</summary>

- **Constructor** — called when an object is created, used to initialize
- **Destructor** — called when an object is destroyed, used to clean up

```cpp
class Foo {
public:
    Foo()           { std::cout << "born"; }    // constructor
    Foo(int x)      { val = x; }                // parameterized constructor
    ~Foo()          { std::cout << "dead"; }    // destructor

private:
    int val;
};

{
    Foo f(42);   // constructor called
}                // destructor called when f goes out of scope
```

</details>

---

**Q24. What are the access specifiers?**

<details><summary>Answer</summary>

- `public` — accessible from anywhere
- `private` — accessible only within the class itself
- `protected` — accessible within the class and derived classes

```cpp
class Base {
public:    int a;    // anyone
protected: int b;    // Base + Derived
private:   int c;    // only Base
};

class Derived : public Base {
    void foo() {
        a = 1;   // ok
        b = 2;   // ok
        c = 3;   // error
    }
};
```

</details>

---

**Q25. What is inheritance?**

<details><summary>Answer</summary>

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
d.eat();    // inherited
d.bark();   // own method
```

Inheritance types: `public` (most common), `protected`, `private` — controls how inherited members are exposed further.

</details>

---

**Q26. What is polymorphism? What is a virtual function?**

<details><summary>Answer</summary>

Polymorphism lets you call the correct method based on the **actual type** at runtime, not the declared type.

```cpp
class Shape {
public:
    virtual void draw() { std::cout << "shape"; }  // virtual!
};

class Circle : public Shape {
public:
    void draw() override { std::cout << "circle"; }
};

Shape* s = new Circle();
s->draw();   // prints "circle" — not "shape"
             // without virtual, would print "shape"
```

- `virtual` — enables runtime dispatch (vtable lookup)
- `override` — tells compiler you're overriding a virtual function (catches typos)
- Without `virtual` — compile-time dispatch, always calls base version

</details>

---

**Q27. What is a pure virtual function? What is an abstract class?**

<details><summary>Answer</summary>

A pure virtual function has no implementation — derived classes **must** override it.

```cpp
class Shape {
public:
    virtual void draw() = 0;   // pure virtual
};

class Circle : public Shape {
public:
    void draw() override { std::cout << "circle"; }
};

Shape s;      // error — cannot instantiate abstract class
Circle c;     // ok
```

A class with at least one pure virtual function is **abstract**. It defines an interface.

</details>

---

**Q28. What is the Rule of Three / Rule of Five?**

<details><summary>Answer</summary>

If you define any of these, you usually need all of them:

**Rule of Three (C++03):**
1. Destructor
2. Copy constructor
3. Copy assignment operator

**Rule of Five (C++11, adds move semantics):**
4. Move constructor
5. Move assignment operator

```cpp
class Buffer {
    int* data;
public:
    Buffer(int size) : data(new int[size]) {}
    ~Buffer() { delete[] data; }                         // 1
    Buffer(const Buffer& other) { /* deep copy */ }      // 2
    Buffer& operator=(const Buffer& other) { ... }       // 3
    Buffer(Buffer&& other) noexcept { /* steal */ }      // 4
    Buffer& operator=(Buffer&& other) noexcept { ... }   // 5
};
```

Needed whenever your class manually manages a resource (heap memory, file handle, socket).

</details>

---

## 7. Memory Management

**Q29. What is the difference between stack and heap memory?**

<details><summary>Answer</summary>

| Feature | Stack | Heap |
|---------|-------|------|
| Allocation | automatic (on scope entry) | manual (`new`) |
| Deallocation | automatic (on scope exit) | manual (`delete`) |
| Size | small (~1-8MB) | large (GBs) |
| Speed | fast | slower |
| Fragmentation | none | possible |

```cpp
int x = 5;              // stack
int* p = new int(5);    // heap
delete p;               // must free manually
```

</details>

---

**Q30. What is a memory leak?**

<details><summary>Answer</summary>

When heap memory is allocated but never freed — the program loses the pointer but the memory stays occupied.

```cpp
void leak() {
    int* p = new int(5);
    return;   // p goes out of scope, but the heap int is never deleted
}
```

Fix with `delete` or (better) smart pointers:
```cpp
void noLeak() {
    auto p = std::make_unique<int>(5);
    return;   // unique_ptr destructor automatically frees memory
}
```

</details>

---

**Q31. What is `new` and `delete`? What about `new[]` and `delete[]`?**

<details><summary>Answer</summary>

```cpp
int* p = new int(42);      // allocate single int
delete p;                  // free it

int* arr = new int[10];    // allocate array of 10 ints
delete[] arr;              // free array — must use delete[], not delete
```

Mismatch (`delete` on array or `delete[]` on single) is **undefined behavior**.

Modern C++: prefer `std::make_unique` / `std::make_shared` over raw `new`.

</details>

---

**Q32. What are smart pointers and why use them?**

<details><summary>Answer</summary>

Smart pointers automatically manage heap memory — no manual `delete` needed.

```cpp
#include <memory>

// unique_ptr — single owner, auto-frees when out of scope
auto p = std::make_unique<int>(42);

// shared_ptr — multiple owners, freed when last owner dies
auto s1 = std::make_shared<int>(42);
auto s2 = s1;   // both own it, ref count = 2

// weak_ptr — non-owning reference, used to break cycles
std::weak_ptr<int> w = s1;
```

| Type | Ownership | Use case |
|------|-----------|----------|
| `unique_ptr` | exclusive | default choice |
| `shared_ptr` | shared | multiple owners |
| `weak_ptr` | none | break shared_ptr cycles |

</details>

---

## 8. Templates

**Q33. What is a function template?**

<details><summary>Answer</summary>

A template lets you write one function that works for multiple types.

```cpp
template <typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

max(3, 5);       // int version
max(3.0, 5.0);   // double version
max('a', 'z');   // char version
```

The compiler generates a concrete function for each type used — this is called **template instantiation**.

</details>

---

**Q34. What is a class template?**

<details><summary>Answer</summary>

A blueprint for a class parameterized by type.

```cpp
template <typename T>
class Stack {
    std::vector<T> data;
public:
    void push(T val) { data.push_back(val); }
    T pop() { T v = data.back(); data.pop_back(); return v; }
    bool empty() { return data.empty(); }
};

Stack<int> s;
s.push(1);
s.push(2);
s.pop();   // 2

Stack<std::string> ss;
ss.push("hello");
```

`std::vector`, `std::map`, etc. are all class templates.

</details>

---

## 9. STL (Standard Template Library)

**Q35. What are the main STL containers?**

<details><summary>Answer</summary>

| Container | Description | Access | Insert/Delete |
|-----------|-------------|--------|--------------|
| `vector` | dynamic array | O(1) random | O(1) end, O(n) middle |
| `deque` | double-ended queue | O(1) random | O(1) both ends |
| `list` | doubly linked list | O(n) | O(1) anywhere (with iterator) |
| `set` | sorted unique elements | O(log n) | O(log n) |
| `map` | sorted key-value pairs | O(log n) | O(log n) |
| `unordered_set` | hash set | O(1) avg | O(1) avg |
| `unordered_map` | hash map | O(1) avg | O(1) avg |
| `stack` | LIFO | top only | O(1) |
| `queue` | FIFO | front/back | O(1) |
| `priority_queue` | max-heap | O(1) top | O(log n) |

</details>

---

**Q36. How does `std::map` work? How is it different from `std::unordered_map`?**

<details><summary>Answer</summary>

```cpp
#include <map>
std::map<std::string, int> m;
m["alice"] = 10;
m["bob"] = 20;
m.count("alice")   // 1 if exists
m.find("bob")      // iterator

#include <unordered_map>
std::unordered_map<std::string, int> um;
um["alice"] = 10;
```

| Feature | `map` | `unordered_map` |
|---------|-------|-----------------|
| Internal structure | Red-black tree | Hash table |
| Key ordering | sorted | none |
| Lookup | O(log n) | O(1) avg |
| Worst case | O(log n) | O(n) |

Use `map` when you need sorted keys. Use `unordered_map` for faster lookups.

</details>

---

**Q37. What are STL algorithms? Give examples.**

<details><summary>Answer</summary>

`#include <algorithm>` gives you a library of generic algorithms that work on iterators.

```cpp
std::vector<int> v = {3, 1, 4, 1, 5, 9};

std::sort(v.begin(), v.end());                  // sort ascending
std::sort(v.begin(), v.end(), std::greater<>());// sort descending

std::find(v.begin(), v.end(), 4);               // iterator to 4
std::count(v.begin(), v.end(), 1);              // 2

std::reverse(v.begin(), v.end());
std::min_element(v.begin(), v.end());           // iterator to min
std::max_element(v.begin(), v.end());

std::for_each(v.begin(), v.end(), [](int x){ std::cout << x; });
```

</details>

---

## 10. Modern C++ (C++11/14/17)

**Q38. What is a lambda expression?**

<details><summary>Answer</summary>

An anonymous (inline) function.

```cpp
auto add = [](int a, int b) { return a + b; };
add(3, 4);   // 7

// capture variables from surrounding scope
int x = 10;
auto addX = [x](int n) { return n + x; };   // capture by value
auto addX2 = [&x](int n) { x += n; };       // capture by reference

// used with algorithms
std::sort(v.begin(), v.end(), [](int a, int b){ return a > b; });
```

Capture list: `[]` nothing, `[=]` all by value, `[&]` all by ref, `[x, &y]` mix.

</details>

---

**Q39. What is `nullptr` and why prefer it over `NULL`?**

<details><summary>Answer</summary>

`nullptr` is a typed null pointer constant (type: `std::nullptr_t`).

```cpp
void foo(int x)   { }
void foo(int* p)  { }

foo(NULL);      // ambiguous — NULL is just 0, could call int version
foo(nullptr);   // unambiguous — always calls pointer version
```

Always prefer `nullptr` in modern C++.

</details>

---

**Q40. What is move semantics? What is `std::move`?**

<details><summary>Answer</summary>

Move semantics lets you **transfer** resources instead of copying them — much cheaper for large objects.

```cpp
std::vector<int> a = {1, 2, 3, 4, 5};
std::vector<int> b = std::move(a);   // a's data is "stolen" by b
// a is now empty, no copy was made
```

- **Copy** — duplicates all data, O(n)
- **Move** — transfers ownership, O(1)

Move is automatically used when returning local variables from functions (NRVO/RVO) and when passing temporaries.

</details>

---

**Q41. What is a range-based for loop?**

<details><summary>Answer</summary>

```cpp
std::vector<int> v = {1, 2, 3};

for (int x : v) { }        // copy each element
for (int& x : v) { }       // reference — can modify
for (const int& x : v) { } // const reference — read-only, no copy

// works on arrays too
int arr[] = {1, 2, 3};
for (int x : arr) { }
```

Prefer `const auto&` for complex types to avoid copying.

</details>

---

**Q42. What is `constexpr`?**

<details><summary>Answer</summary>

`constexpr` means the value or function can be evaluated **at compile time**.

```cpp
constexpr int square(int x) { return x * x; }

constexpr int val = square(5);   // computed at compile time: 25
int arr[square(3)];              // ok — array size must be compile-time constant
```

Difference from `const`:
- `const` — value is fixed at runtime
- `constexpr` — value is fixed at compile time

</details>

---

## 11. Exception Handling

**Q43. How does exception handling work in C++?**

<details><summary>Answer</summary>

```cpp
try {
    int x = -1;
    if (x < 0) throw std::invalid_argument("negative value");
    // risky code
} catch (const std::invalid_argument& e) {
    std::cout << "caught: " << e.what();
} catch (const std::exception& e) {
    std::cout << "general exception: " << e.what();
} catch (...) {
    std::cout << "unknown exception";
}
```

- `throw` — raises an exception
- `try` — block where exceptions might be thrown
- `catch` — handles specific exception types

</details>

---

**Q44. What is `noexcept`?**

<details><summary>Answer</summary>

Marks a function as guaranteed to not throw exceptions.

```cpp
void foo() noexcept {
    // if this throws anyway, std::terminate is called
}
```

Benefits:
- Enables compiler optimizations
- Required for move constructors in some STL operations
- Documents intent

</details>

---

## 12. Miscellaneous

**Q45. What is the difference between `++i` and `i++`?**

<details><summary>Answer</summary>

Both increment `i`, but they return different values:
- `++i` — pre-increment: increments first, **returns new value**
- `i++` — post-increment: **returns old value**, then increments

```cpp
int i = 5;
int a = ++i;   // i is 6, a is 6
int b = i++;   // b is 6, then i becomes 7
```

For iterators and complex types, `++i` is preferred (no temporary copy needed).

</details>

---

**Q46. What is operator overloading?**

<details><summary>Answer</summary>

Defining custom behavior for operators on your own types.

```cpp
class Vec2 {
public:
    float x, y;
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    bool operator==(const Vec2& other) const {
        return x == other.x && y == other.y;
    }
};

Vec2 a(1, 2), b(3, 4);
Vec2 c = a + b;   // calls operator+
```

Common overloadable operators: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `[]`, `()`, `<<`, `>>`.

</details>

---

**Q47. What is `static` inside a class vs inside a function?**

<details><summary>Answer</summary>

**Static class member** — shared across all instances:
```cpp
class Counter {
public:
    static int count;           // one copy for all instances
    Counter() { count++; }
};
int Counter::count = 0;         // must define outside class

Counter a, b, c;
Counter::count;   // 3
```

**Static local variable** — initialized once, persists across calls:
```cpp
int nextId() {
    static int id = 0;   // initialized only the first time
    return ++id;
}
nextId();   // 1
nextId();   // 2
nextId();   // 3
```

</details>

---

**Q48. What is the difference between `struct` initialization styles?**

<details><summary>Answer</summary>

```cpp
struct Point { int x, y; };

Point p1;            // uninitialized (undefined values)
Point p2 = {1, 2};  // aggregate initialization
Point p3{1, 2};      // uniform initialization (C++11, preferred)
Point p4 = {};       // zero-initialized: {0, 0}
Point p5{};          // same as above
```

Uniform initialization (`{}`) is preferred in modern C++ — it works everywhere and prevents narrowing conversions.

</details>

---

## Quick Reference Cheat Sheet

```
Types:        int, double, float, char, bool, long long, auto
Pointers:     int* p = &x;   *p (deref)   p->member
References:   int& r = x;    r is an alias
Const:        const int* (data const)   int* const (ptr const)
Memory:       new/delete   new[]/delete[]   prefer smart ptrs
RAII:         unique_ptr, shared_ptr, weak_ptr
Containers:   vector, map, set, unordered_map, stack, queue
Algorithms:   sort, find, count, reverse, for_each
Lambdas:      [capture](params) { body }
Templates:    template<typename T> T func(T a) { }
OOP:          class, public/private/protected, virtual, override
Exceptions:   try { } catch (const std::exception& e) { }
Modern:       nullptr, auto, constexpr, noexcept, std::move
```
