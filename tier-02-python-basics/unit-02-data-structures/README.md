# Unit 2 — Data Structures

## Topics

### 1. Lists
- Ordered, mutable, allow duplicates
- Indexing and slicing: `lst[0]`, `lst[-1]`, `lst[1:4]`, `lst[::2]`
- Methods: `append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`
- Nested lists (2-D grids)

### 2. Tuples
- Ordered, **immutable**, allow duplicates
- Use for fixed data (coordinates, RGB values, dict keys)
- Tuple unpacking: `a, b, c = (1, 2, 3)`

### 3. Dictionaries
- Unordered (insertion-ordered in Python 3.7+), mutable key-value pairs
- Keys must be hashable (strings, numbers, tuples)
- Methods: `keys()`, `values()`, `items()`, `get()`, `setdefault()`
- `defaultdict` and `Counter` from `collections`

### 4. Sets
- Unordered, no duplicates
- Fast membership testing: `x in s` is O(1)
- Set operations: union (`|`), intersection (`&`), difference (`-`), symmetric difference (`^`)

### 5. List Comprehensions
- Concise syntax: `[expr for item in iterable if condition]`
- Dict/set comprehensions: `{k: v for k, v in ...}`
- Generator expressions (lazy evaluation): `(expr for item in ...)`

### 6. Iterators and Generators
- `iter()`, `next()`
- `yield` keyword: generator functions
- Built-ins: `enumerate()`, `zip()`, `map()`, `filter()`

---

## Exercises

### Exercise 1 — List Operations
Given the list `scores = [88, 72, 95, 60, 78, 95, 88]`:
1. Get the top 3 unique scores
2. Count how many scores are above 80
3. Replace all scores below 70 with 70 (floor at 70)

```python
scores = [88, 72, 95, 60, 78, 95, 88]

top3 = sorted(set(scores), reverse=True)[:3]
above_80 = sum(1 for s in scores if s > 80)
floored = [max(s, 70) for s in scores]

print("Top 3 unique:", top3)
print("Above 80:", above_80)
print("Floored:", floored)
```

### Exercise 2 — Dictionary Operations
Build a word frequency counter from a sentence without using `Counter`:

```python
sentence = "the cat sat on the mat the cat"
freq = {}
for word in sentence.split():
    freq[word] = freq.get(word, 0) + 1
print(freq)
# Verify with Counter:
from collections import Counter
print(Counter(sentence.split()))
```

### Exercise 3 — Set Operations
Given two lists of student names in two classes, find:
1. Students in both classes (intersection)
2. Students in class A but not class B (difference)
3. Students in either class (union)

```python
class_a = {"Alice", "Bob", "Charlie", "Diana"}
class_b = {"Bob", "Diana", "Eve", "Frank"}
print("Both:", class_a & class_b)
print("Only A:", class_a - class_b)
print("Either:", class_a | class_b)
```

### Exercise 4 — Comprehensions
1. Create a list of squares for even numbers 0–20
2. Build a dict mapping each word in a sentence to its length
3. Write a generator that yields Fibonacci numbers up to a limit

```python
# 1. Squares of evens
squares = [x**2 for x in range(0, 21) if x % 2 == 0]
print(squares)

# 2. Word lengths
sentence = "machine learning is powerful"
lengths = {word: len(word) for word in sentence.split()}
print(lengths)

# 3. Fibonacci generator
def fibonacci(limit):
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b
print(list(fibonacci(100)))
```

---

## Questions & Answers

**Q1: When should I use a list vs a tuple?**
> Use a tuple for data that should not change (function return values returning multiple items, coordinates, dictionary keys). Use a list when you need a mutable, ordered sequence of items you will modify (append, sort, etc.). Tuples are also slightly faster than lists and can be used as dictionary keys.

**Q2: Why is dictionary lookup O(1)?**
> Dictionaries are implemented as hash tables. Python hashes the key to compute an index into an internal array. On average, this is a constant-time operation regardless of how many entries the dictionary has (though worst-case with hash collisions is O(n)).

**Q3: Why prefer list comprehensions over for loops?**
> List comprehensions are more Pythonic, often faster (the looping happens at the C level inside Python), and more readable for simple transformations. However, if the logic is complex or you need nested loops, a for loop may be clearer.

**Q4: What is the difference between `map()` and a list comprehension?**
> Both apply a function element-wise to an iterable. `map()` returns a lazy iterator; you need `list(map(...))` to materialize it. List comprehensions are generally preferred in Python for readability. `map()` can be marginally faster when calling an existing built-in or named function.

**Q5: What is the advantage of a generator over a list?**
> A generator produces values lazily one at a time, consuming O(1) memory regardless of how many values it will produce. A list stores all values in memory at once. Generators are essential when working with large datasets or infinite sequences.
