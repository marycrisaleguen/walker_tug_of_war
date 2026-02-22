# Unit 1 — Python Syntax

## Topics

### 1. Variables and Data Types
- Integers, floats, strings, booleans
- Dynamic typing — no declaration needed
- Type checking: `type()`, `isinstance()`

### 2. Control Flow
- `if / elif / else`
- `for` loops over iterables
- `while` loops
- `break`, `continue`, `pass`

### 3. Functions
- `def` keyword, parameters, return values
- Default arguments, keyword arguments, `*args`, `**kwargs`
- Scope: local vs global variables
- Lambda functions (anonymous)
- Docstrings

### 4. Modules and Packages
- `import module`, `from module import name`
- Standard library highlights: `math`, `random`, `os`, `sys`, `collections`
- Virtual environments: `venv`, `pip`

### 5. File I/O
- `open()`, `read()`, `write()`, `close()`
- Context managers: `with open(...) as f:`
- Reading CSV manually vs with `pandas`

---

## Exercises

### Exercise 1 — FizzBuzz
Print numbers 1–50. Print "Fizz" for multiples of 3, "Buzz" for multiples of 5, and "FizzBuzz" for multiples of both.

```python
for n in range(1, 51):
    if n % 15 == 0:
        print("FizzBuzz")
    elif n % 3 == 0:
        print("Fizz")
    elif n % 5 == 0:
        print("Buzz")
    else:
        print(n)
```

### Exercise 2 — Functions
Write a function `describe_number(n)` that returns:
- `"negative"` if n < 0
- `"zero"` if n == 0
- `"small positive"` if 0 < n < 10
- `"large positive"` if n >= 10

```python
def describe_number(n):
    """Classify n into one of four categories."""
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    elif n < 10:
        return "small positive"
    else:
        return "large positive"

for test in [-5, 0, 3, 42]:
    print(f"{test}: {describe_number(test)}")
```

### Exercise 3 — *args and **kwargs
Write a function `stats(*numbers)` that accepts any number of arguments and prints their min, max, and mean.

```python
def stats(*numbers):
    print(f"Min: {min(numbers)}, Max: {max(numbers)}, Mean: {sum(numbers)/len(numbers):.2f}")

stats(4, 7, 2, 9, 1)
```

### Exercise 4 — File I/O
Write a Python script that writes the numbers 1–10 (one per line) to a file `numbers.txt`, then reads it back and prints the sum.

```python
with open("/tmp/numbers.txt", "w") as f:
    for i in range(1, 11):
        f.write(f"{i}\n")

total = 0
with open("/tmp/numbers.txt") as f:
    for line in f:
        total += int(line.strip())
print(f"Sum: {total}")  # Expected: 55
```

---

## Questions & Answers

**Q1: What does "dynamically typed" mean in Python?**
> Python determines variable types at runtime rather than at compile time. You do not declare types — a variable can hold a string, then later be reassigned to an integer. The `type()` function reveals the current type. This makes Python flexible but also means type errors can appear at runtime rather than being caught early.

**Q2: What is the difference between `==` and `is`?**
> `==` checks value equality (do the two objects have the same value?). `is` checks identity (do both variables point to the exact same object in memory?). Always use `==` for comparing values. Use `is` only for `None` checks: `if x is None`.

**Q3: What is a lambda function and when should I use it?**
> A lambda is a one-line anonymous function: `lambda x: x**2`. Use lambdas for short, simple functions passed as arguments (e.g., `sorted(data, key=lambda x: x[1])`). For anything longer or reused, define a named function with `def` for readability.

**Q4: What is the difference between a module and a package?**
> A module is a single `.py` file. A package is a directory containing an `__init__.py` file and multiple modules. When you `import numpy`, you are importing a package. When you `import math`, you are importing a standard library module.

**Q5: Why use a `with` statement for file handling?**
> The `with` statement ensures the file is properly closed even if an exception is raised. It is equivalent to a try/finally block that calls `f.close()`. Always use `with open(...)` to avoid resource leaks.
