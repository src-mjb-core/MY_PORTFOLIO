## Row Picture and Column Picture — A Geometric View of Linear Systems

This computational exploration is inspired by **Lecture 1: “The Geometry of Linear Equations”** and is designed to build intuition for solving linear systems by *seeing* them geometrically.

The program visualizes **row picture** and **column picture** side by side, helping learners understand how the same system of equations can be interpreted in two fundamentally different—but equivalent—ways.

The emphasis here is on *understanding*, not on algorithmic efficiency or original code authorship.

---

## The Fundamental Problem

The central problem of linear algebra is to solve systems of linear equations written in matrix form:

\[
A x = b
\]

Here:
- \( A \) is the matrix of coefficients
- \( x \) is the vector of unknowns
- \( b \) is the right-hand-side vector

This program focuses on the geometric meaning of this equation.

---

## Three Complementary Viewpoints

### 1. Row Picture

In the **row picture**, each row of the matrix \( A \) represents one equation.

- In **2 variables**, each equation is a **line** in the \(x\)-\(y\) plane.
- In **3 variables**, each equation is a **plane** in 3D space.
- The solution corresponds to the **intersection** of all lines or planes.

Geometric outcomes:
- **Unique solution** → all objects intersect at a single point
- **No solution** → parallel or inconsistent geometry
- **Infinite solutions** → overlapping or dependent geometry

This viewpoint aligns closely with elimination and consistency of equations.

---

### 2. Column Picture

In the **column picture**, the same system is interpreted as a vector equation:

**x₁a₁ + x₂a₂ + … + xₙaₙ = b**

where a₁, a₂...… aₙ are the **columns of \(A\)**.

From this perspective:
- The unknowns \(x_i\) are **weights**
- The question becomes:  
  *Can vector \(b\) be formed as a linear combination of the columns of \(A\)?*

This view naturally introduces:
- span
- linear combinations
- vector spaces
- rank and dimension

---

### 3. Matrix Picture

The matrix form \(A x = b\) compactly encodes all equations at once and is the foundation for computational methods such as Gaussian elimination.

Each **row** corresponds to one equation.  
Each **column** corresponds to one variable across all equations.

---

## What This Program Does

This program allows the user to:

- Choose the number of variables (from 1 up to 6)
- Input a square system of linear equations
- View **row picture** and **column picture simultaneously**
- Observe how the same system behaves geometrically across dimensions

For dimensions higher than 3:
- The **row picture** is shown as a 3D slice of higher-dimensional hyperplanes
- The **column picture** is shown as a 3D projection (shadow) of vector combinations

---

## Why Both Pictures Matter

The **row picture** becomes difficult to interpret clearly beyond three dimensions, since intersections of hyperplanes cannot be visualized directly.

The **column picture**, however, scales naturally to higher dimensions:
- It represents solutions as vector combinations
- It allows projection of high-dimensional behavior into understandable geometric paths

For this reason, the column picture plays a crucial role in understanding higher-dimensional systems, even when row-based geometry becomes abstract.

---

## About the Code

This code was developed with support from external references and tools.

The purpose of the *computational explorations* in this repository is **not** to demonstrate original programming, but to create clear, working visualizations that help learners understand mathematical concepts that are otherwise difficult to see.

The focus is on:
- intuition
- clarity
- conceptual understanding

If this program helps you visualize how equations interact in 2D, 3D, or higher-dimensional systems, then it has achieved its goal.

---

## Limitations

- Higher-dimensional systems are shown via projection, not exact geometry
- Visual accuracy decreases as dimension increases
- The goal is insight, not precision

---

## How to Use

Run the Python script and follow the prompts in the terminal.  
The program will guide you through input and display both geometric interpretations side by side.
