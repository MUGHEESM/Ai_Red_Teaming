# Mathematics Refresher for AI

## Overview

This module delves into some mathematical concepts behind AI algorithms and processes. If you come across symbols or notations that are unfamiliar, feel free to refer back to this page for a quick refresher. You don't need to understand everything here; it's primarily meant to serve as a reference.

---

## Basic Arithmetic Operations

### Multiplication (*)
The multiplication operator denotes the product of two numbers or expressions.

```python
3 * 4 = 12
```

### Division (/)
The division operator denotes dividing one number or expression by another.

```python
10 / 2 = 5
```

### Addition (+)
The addition operator represents the sum of two or more numbers or expressions.

```python
5 + 3 = 8
```

### Subtraction (-)
The subtraction operator represents the difference between two numbers or expressions.

```python
9 - 4 = 5
```

---

## Algebraic Notations

### Subscript Notation (x_t)
The subscript notation represents a variable indexed by t, often indicating a specific time step or state in a sequence.

```python
x_t = q(x_t | x_{t-2})
```

This notation is commonly used in sequences and time series data, where each `x_t` represents the value of x at time t.

### Superscript Notation (x^n)
Superscript notation is used to denote exponents or powers.

```python
x^2 = x * x
```

This notation is used in polynomial expressions and exponential functions.

### Norm (||...||)
The norm measures the size or length of a vector. The most common norm is the Euclidean norm:

```python
||v|| = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
```

Other norms include:
- **L1 norm (Manhattan distance)**:
  ```python
  ||v||_1 = |v_1| + |v_2| + ... + |v_n|
  ```
- **L∞ norm (maximum absolute value)**:
  ```python
  ||v||_∞ = max(|v_1|, |v_2|, ..., |v_n|)
  ```

Norms are used in measuring distances between vectors, regularizing models, and normalizing data.

### Summation Symbol (Σ)
The summation symbol indicates the sum of a sequence of terms.

```python
Σ_{i=1}^{n} a_i
```

This represents the sum of the terms a_1, a_2, ..., a_n.

---

## Logarithms and Exponentials

### Logarithm Base 2 (log2(x))
The logarithm base 2 is often used in information theory to measure entropy.

```python
log2(8) = 3
```

### Natural Logarithm (ln(x))
The natural logarithm is the logarithm with base e (Euler's number).

```python
ln(e^2) = 2
```

Widely used in calculus, differential equations, and probability theory.

### Exponential Function (e^x)
Represents Euler's number e raised to the power of x.

```python
e^2 ≈ 7.389
```

Used to model growth and decay processes, probability distributions, and various mathematical models.

### Exponential Function Base 2 (2^x)
Represents 2 raised to the power of x, often used in binary systems.

```python
2^3 = 8
```

---

## Matrix and Vector Operations

### Matrix-Vector Multiplication (A * v)
The product of a matrix A and a vector v.

```python
A * v = [[1, 2], [3, 4]] * [5, 6] = [17, 39]
```

Fundamental in linear algebra and neural networks.

### Matrix-Matrix Multiplication (A * B)
The product of two matrices A and B.

```python
A * B = [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
```

### Transpose (A^T)
Swaps the rows and columns of matrix A.

```python
A = [[1, 2], [3, 4]]
A^T = [[1, 3], [2, 4]]
```

### Inverse (A^{-1})
The matrix that, when multiplied by A, results in the identity matrix.

```python
A = [[1, 2], [3, 4]]
A^{-1} = [[-2, 1], [1.5, -0.5]]
```

Used to solve systems of linear equations.

### Determinant (det(A))
A scalar value computed from a square matrix.

```python
A = [[1, 2], [3, 4]]
det(A) = 1 * 4 - 2 * 3 = -2
```

Determines whether a matrix is invertible (non-zero determinant).

### Trace (tr(A))
The sum of the elements on the main diagonal.

```python
A = [[1, 2], [3, 4]]
tr(A) = 1 + 4 = 5
```

---

## Set Theory

### Cardinality (|S|)
The number of elements in a set S.

```python
S = {1, 2, 3, 4, 5}
|S| = 5
```

### Union (∪)
The set of all elements in either set A or B or both.

```python
A = {1, 2, 3}, B = {3, 4, 5}
A ∪ B = {1, 2, 3, 4, 5}
```

### Intersection (∩)
The set of all elements in both A and B.

```python
A = {1, 2, 3}, B = {3, 4, 5}
A ∩ B = {3}
```

### Complement (A^c)
The set of all elements not in A.

```python
U = {1, 2, 3, 4, 5}, A = {1, 2, 3}
A^c = {4, 5}
```

---

## Comparison Operators

### Greater Than or Equal To (>=)
```python
a >= b
```

### Less Than or Equal To (<=)
```python
a <= b
```

### Equality (==)
```python
a == b
```

### Inequality (!=)
```python
a != b
```

---

## Eigenvalues and Eigenvectors

### Lambda (Eigenvalue) λ
Represents an eigenvalue in linear algebra.

```python
A * v = λ * v, where λ = 3
```

Used in understanding linear transformations, PCA, and optimization.

### Eigenvector
A non-zero vector that, when multiplied by a matrix, results in a scalar multiple of itself.

```python
A * v = λ * v
```

Used in dimensionality reduction techniques like PCA.

---

## Functions and Operators

### Maximum Function (max(...))
Returns the largest value from a set of values.

```python
max(4, 7, 2) = 7
```

### Minimum Function (min(...))
Returns the smallest value from a set of values.

```python
min(4, 7, 2) = 2
```

### Reciprocal (1 / ...)
One divided by an expression, inverting the value.

```python
1 / x where x = 5 results in 0.2
```

### Ellipsis (...)
Indicates the continuation of a pattern or sequence.

```python
a_1 + a_2 + ... + a_n
```

---

## Functions and Probability

### Function Notation (f(x))
Represents a function f applied to an input x.

```python
f(x) = x^2 + 2x + 1
```

### Conditional Probability Distribution (P(x | y))
The probability distribution of x given y.

```python
P(Output | Input)
```

Used in Bayesian inference and probabilistic models.

### Expectation Operator (E[...])
The expected value or average of a random variable.

```python
E[X] = Σ x_i * P(x_i)
```

### Variance (Var(X))
Measures the spread of a random variable around its mean.

```python
Var(X) = E[(X - E[X])^2]
```

### Standard Deviation (σ(X))
The square root of the variance.

```python
σ(X) = sqrt(Var(X))
```

### Covariance (Cov(X, Y))
Measures how two random variables vary together.

```python
Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
```

### Correlation (ρ(X, Y))
A normalized measure of covariance, ranging from -1 to 1.

```python
ρ(X, Y) = Cov(X, Y) / (σ(X) * σ(Y))
```

Indicates the strength and direction of the linear relationship between two variables.

---

## Quick Reference Summary

This page serves as a mathematical reference for the AI concepts covered throughout this module. Refer back to specific sections as needed when encountering unfamiliar notation in the course material.

