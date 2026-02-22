# Unit 3 — Object-Oriented Programming (OOP)

## Topics

### 1. Classes and Instances
- `class` keyword, `__init__` constructor
- Instance attributes (`self.x`) vs class attributes
- Creating instances: `obj = MyClass(args)`

### 2. Methods
- Instance methods: take `self` as the first parameter
- Class methods: `@classmethod`, take `cls`
- Static methods: `@staticmethod`, no implicit first argument

### 3. Inheritance
- Subclass inherits attributes and methods from superclass
- `super().__init__()` to call the parent constructor
- Method overriding

### 4. Encapsulation
- Convention: `_protected` and `__private` name mangling
- Properties: `@property`, `@attr.setter`

### 5. Magic Methods (Dunder Methods)
- `__str__`, `__repr__`: string representation
- `__len__`, `__getitem__`: behave like a container
- `__eq__`, `__lt__`: comparison operators
- `__add__`, `__mul__`: arithmetic operators

### 6. Practical OOP in ML
- Scikit-learn API: `fit()`, `transform()`, `predict()`
- Custom transformers inheriting from `BaseEstimator`, `TransformerMixin`

---

## Exercises

### Exercise 1 — Define a Class
Create a `Vector2D` class representing a 2-D vector with:
- `__init__(self, x, y)`
- `__repr__`: display as `Vector2D(x, y)`
- `__add__`: vector addition
- `magnitude()`: return ‖v‖₂
- `dot(other)`: dot product

```python
import math

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

v1 = Vector2D(3, 4)
v2 = Vector2D(1, -1)
print(v1 + v2)             # Vector2D(4, 3)
print(v1.magnitude())      # 5.0
print(v1.dot(v2))          # -1
```

### Exercise 2 — Inheritance
Create an `Animal` base class and `Dog` / `Cat` subclasses:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement speak()")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

class Dog(Animal):
    def speak(self):
        return f"{self.name} says: Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says: Meow!"

animals = [Dog("Rex"), Cat("Whiskers"), Dog("Buddy")]
for a in animals:
    print(a.speak())
```

### Exercise 3 — Custom sklearn Transformer
Build a custom `Log1pTransformer` that applies log(1+x) to features, compatible with scikit-learn pipelines:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        return np.log1p(X)

# Test it
import numpy as np
X = np.array([[0, 1, 100], [2, 9, 1000]])
t = Log1pTransformer()
print(t.fit_transform(X))
```

---

## Questions & Answers

**Q1: What is `self` in Python?**
> `self` is a reference to the current instance of the class. It is the first parameter of every instance method and is passed automatically when you call the method on an object. It allows methods to access and modify the instance's attributes.

**Q2: What is the difference between `__str__` and `__repr__`?**
> `__repr__` should return an unambiguous, developer-facing string (ideally one that can be passed to `eval()` to recreate the object). `__str__` should return a human-readable string. `print(obj)` calls `__str__`; the Python REPL and `repr(obj)` call `__repr__`. If only `__repr__` is defined, it is used for both.

**Q3: What is the benefit of inheriting from `BaseEstimator` and `TransformerMixin` in sklearn?**
> `BaseEstimator` provides `get_params()` and `set_params()` methods, enabling your transformer to work with `GridSearchCV` and cloning utilities. `TransformerMixin` provides a default `fit_transform()` method. Together they make your custom transformer fully compatible with sklearn pipelines.

**Q4: When should I use `@classmethod` vs `@staticmethod`?**
> Use `@classmethod` when the method needs access to the class itself (e.g., factory methods, alternative constructors). Use `@staticmethod` when the method is logically related to the class but does not need access to the class or instance — it is just a utility function namespaced inside the class.

**Q5: What is duck typing?**
> Python does not require explicit interface declarations. If an object has the methods and attributes required by a given context, it will work — regardless of its class. "If it walks like a duck and quacks like a duck, it's a duck." This is why sklearn just calls `fit()` and `transform()` on any compatible object without checking its type.
