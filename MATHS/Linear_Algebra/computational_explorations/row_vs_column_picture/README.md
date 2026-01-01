## Row Picture vs Column Picture (Linear Algebra)

This computational exploration is designed to help visualize one of the most important conceptual ideas in Linear Algebra:  
**the difference between the row picture and the column picture of a system of linear equations**.

The goal of this program is *not* to showcase advanced programming skills, but to support conceptual understanding through clear geometric visualization.

---

## Conceptual Background

Consider a system of linear equations written in matrix form:

A x = b

Depending on how we interpret this system, we get two complementary geometric viewpoints.

---

### Row Picture

In the **row picture**, each equation is viewed as a geometric object:

- In 2 variables:  
  Each equation represents a **line** in the x–y plane.
- In 3 variables:  
  Each equation represents a **plane** in 3D space.

From this perspective:
- A **unique solution** corresponds to lines (or planes) intersecting at a single point.
- **Infinite solutions** occur when lines coincide or planes intersect along a line.
- **No solution** occurs when lines are parallel or planes do not share a common intersection.

This interpretation aligns closely with ideas from elimination and consistency of equations.

---

### Column Picture

In the **column picture**, the same system is interpreted differently.

Instead of focusing on equations, we focus on **vectors**:
- Each column of the matrix A is treated as a vector.
- The solution represents a **linear combination of these column vectors** that produces the vector b.

Geometrically:
- The question becomes whether vector **b** lies in the span of the columns of A.
- The solution variables act as weights that scale and combine the column vectors.

This viewpoint is essential for understanding concepts such as:
- vector spaces
- span
- linear combinations
- rank and dimension

---

## What This Program Does

This program allows the user to:

- Choose the number of equations (1, 2, or 3)
- Enter the coefficients of the equations
- Select whether to visualize:
  - the **row picture**, or
  - the **column picture**
- See how the same linear system behaves geometrically in 2D or 3D

The visualizations are intentionally kept simple so that the geometric intuition is clear and not hidden behind technical complexity.

---

## Relation to Lecture Concepts

The ideas visualized here connect directly to standard Linear Algebra theory, including:

- Conditions for **unique**, **infinite**, or **no solutions**
- Intersection of lines and planes
- Consistency of linear systems
- Interpretation of solutions through geometry rather than algebra alone

These concepts were introduced in foundational Linear Algebra lectures and are explored here visually to reinforce understanding.

---

## About the Code

This code was developed with support from external references and tools.  
The purpose of the *computational explorations* folder is **not** to claim originality of implementation, but to create clear, working visual tools that help learners understand mathematical ideas that are often difficult to visualize.

The emphasis is on:
- clarity
- intuition
- educational value

If this program helps you *see* how equations interact in the x–y or x–y–z space, then it has served its purpose.

---

## Limitations

- Visualizations are restricted to 2D and 3D
- Higher-dimensional systems cannot be displayed geometrically
- The program prioritizes intuition over mathematical completeness

---

## How to Use

Run the Python file and follow the prompts in the terminal.  
You will be guided through equation input and visualization selection step by step.
