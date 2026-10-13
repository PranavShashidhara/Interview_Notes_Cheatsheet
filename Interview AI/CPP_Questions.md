# C++ Practice Questions

Answer each question from memory, then check `CPP_Answers.md`.

---

## 1. Basics

**Q1.** What are the fundamental data types in C++? List at least 6 with their typical sizes.

**Q2.** What is the difference between `int`, `unsigned int`, `long`, and `long long`?

**Q3.** What is `auto` and when should you use it? When should you avoid it?

**Q4.** What is the difference between `=`, `==`, and `===` in C++?

**Q5.** What is `const` and why use it?

---

## 2. Control Flow

**Q6.** What are the 4 loop types in C++ and when do you use each?

**Q7.** What is the difference between `break` and `continue`? Give a code example.

**Q8.** What is a switch statement? What is fall-through and when is it a bug?

---

## 3. Functions

**Q9.** What is function overloading? What can you NOT use to differentiate overloads?

**Q10.** What is the difference between pass-by-value, pass-by-reference, and pass-by-pointer? When would you use each?

**Q11.** What are default arguments? What rule must they follow?

**Q12.** What is recursion? What happens if you forget the base case?

---

## 4. Pointers & References

**Q13.** What is a pointer? Write code to declare one, assign it, and dereference it.

**Q14.** What is a null pointer? What happens if you dereference one?

**Q15.** What is the difference between a pointer and a reference? Give at least 4 differences.

**Q16.** What do each of these mean? `const int* p` — `int* const p` — `const int* const p`

**Q17.** What is pointer arithmetic? What does `p + 1` actually do in memory?

**Q18.** What is a double pointer (`**`)? When do you need one?

---

## 5. Arrays & Strings

**Q19.** How do you declare and initialize a C-style array? What information does it NOT carry?

**Q20.** What is the difference between a C-string and `std::string`? Which should you prefer?

**Q21.** What is `std::vector`? Give 3 differences between it and a raw array.

---

## 6. Object-Oriented Programming

**Q22.** What is the difference between a `class` and a `struct` in C++?

**Q23.** What is a constructor? What is a destructor? When is each called?

**Q24.** What are the 3 access specifiers and what does each allow?

**Q25.** What is inheritance? Write a simple example.

**Q26.** What is polymorphism? What does `virtual` do and what happens without it?

**Q27.** What is a pure virtual function? What is an abstract class?

**Q28.** What is the Rule of Three? What does it expand to in C++11 (Rule of Five)?

---

## 7. Memory Management

**Q29.** What is the difference between stack and heap memory? Give 4 differences.

**Q30.** What is a memory leak? Write a code example and show how to fix it.

**Q31.** What is the difference between `new`/`delete` and `new[]`/`delete[]`? What happens if you mismatch them?

**Q32.** What are the 3 smart pointer types? What is each used for?

---

## 8. Templates

**Q33.** What is a function template? Write one for a `max` function.

**Q34.** What is a class template? Write a simple `Stack<T>` example.

---

## 9. STL

**Q35.** Name at least 8 STL containers and their use cases.

**Q36.** What is the difference between `std::map` and `std::unordered_map`? When do you pick each?

**Q37.** Name 6 STL algorithms from `<algorithm>` and what they do.

---

## 10. Modern C++ (C++11/14/17)

**Q38.** What is a lambda expression? Write one that captures a variable by value and one by reference.

**Q39.** Why prefer `nullptr` over `NULL`? Give an example where it matters.

**Q40.** What is move semantics? What does `std::move` actually do?

**Q41.** What is a range-based for loop? What is the best way to iterate without copying?

**Q42.** What is `constexpr`? How does it differ from `const`?

---

## 11. Exception Handling

**Q43.** Write a try/catch block. What does `e.what()` return?

**Q44.** What does `noexcept` do? What happens if a `noexcept` function throws?

---

## 12. Miscellaneous

**Q45.** What is the difference between `++i` and `i++`? Which is preferred for iterators and why?

**Q46.** What is operator overloading? Write an example for `operator+` on a custom type.

**Q47.** What does `static` mean inside a class? Inside a function? Give examples of both.

**Q48.** What are the different ways to initialize a struct in C++? Which is preferred in modern C++?

---

## Rapid Fire — Answer in one line each

1. What does `&` mean in a declaration vs in an expression?
2. What does `*` mean in a declaration vs in an expression?
3. What is the size of `bool` in C++?
4. Can a reference be null?
5. Can a pointer be reseated?
6. What is the default access in a `class`? In a `struct`?
7. What does `override` do?
8. What is a vtable?
9. What does `delete[]` do differently than `delete`?
10. What is RAII?
