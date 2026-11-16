# Construction Costs Through Linear Algebra

**Mathematics Project · Linear Algebra · 3rd Semester**

A linear-algebra walkthrough of construction cost data.  
We treat monthly series for **cement**, **steel**, **lumber** and **construction wages** as columns of a data matrix and apply the full Unit-1 / Unit-2 toolkit: linear systems, rank, four fundamental subspaces, linear transformations, projections, eigenvalues, positive-definiteness, SVD and PCA.

**Presented by**  
Sanjey · Shabrish · Sridatta · Rohan · Rohit

---

## What the project does

Construction budgets are driven by four measurable inputs:

| Series              | Unit              | Role                                      |
|---------------------|-------------------|-------------------------------------------|
| `cement_index`      | index points      | Binder in concrete, slabs, foundations    |
| `steel_index`       | index points      | Rebar & structural sections               |
| `lumber_index`      | index points      | Formwork, framing, finishing              |
| `construction_wage` | currency / hour   | Labour cost of the trade workers          |

Because three series are unitless indices and one is a wage, raw numbers cannot be compared. We therefore:

1. Build the data matrix \(X\) (months × 4 series)
2. Centre and standardise each column → \(Z\)
3. Form the covariance / correlation matrix \(S\)
4. Define a **basket** linear map \(T:\mathbb{R}^4\to\mathbb{R}^2\)
5. Compute the SVD of \(Z\) and the associated PCA

The basket itself is the linear combination

\[
q = w_1\cdot\text{cement} + w_2\cdot\text{steel} + w_3\cdot\text{lumber},
\qquad w_i\ge 0,\quad\sum w_i=1.
\]

The map is realised by the matrix

\[
A = \begin{bmatrix}
w_1 & w_2 & w_3 & 0 \\
0   & 0   & 0   & 1
\end{bmatrix},
\qquad
T(x)=Ax=(q,\text{wage}).
\]

---

## Syllabus map

| Slides | Topic                                      | Construction link                                      |
|--------|--------------------------------------------|--------------------------------------------------------|
| 3–4    | Gaussian elimination, LU                   | Solving \(Ax=b\) for material unit prices              |
| 5      | Rank, consistency, inverse                 | Independent information in the four series             |
| 6–7    | Four fundamental subspaces, basis, dim     | Column space of the data matrix                        |
| 8      | Linear transformations                     | The basket map \(T\) and its kernel                    |
| 9      | Dot product, norms, Gram–Schmidt           | Correlation = cosine of angle between centred series   |
| 10     | Orthogonal projection & least squares      | Best linear combination of materials that matches wage |
| 11     | Eigenvalues & diagonalisation              | Directions stretched by the covariance operator        |
| 12     | Covariance & positive-definiteness         | \(S\) is always PSD; PD iff columns independent        |
| 13     | Singular-value decomposition               | \(Z=U\Sigma V^T\); link \(\lambda_i=\sigma_i^2/(n-1)\) |
| 14     | Principal-component analysis               | Feature extraction & dimensionality reduction          |

---

## Repository contents

```
LICENSE
numoy.py          # main NumPy implementation of the pipeline
README.md         # this file
```

> **Note on the filename.** The script is named `numoy.py` (a small typo for the intended NumPy module). It implements the six-step plan listed above.

---

## Quick start

```bash
# requirements
pip install numpy pandas matplotlib

# run the analysis (expects a CSV with the four series)
python numoy.py
```

### Expected data format

- One row = one calendar month  
- At least 24 consecutive months from the **same** market  
- Columns: `cement_index`, `steel_index`, `lumber_index`, `construction_wage`  
- Real, cited data only (no synthetic numbers)

The script will:

- standardise the columns,
- form the covariance matrix and verify positive-definiteness,
- apply the basket transformation,
- compute the SVD / PCA,
- print loadings, explained-variance ratios and the first few principal scores.

---

## Mathematical highlights

**Covariance under a linear map**

\[
\operatorname{Cov}(Ax) = A\,S\,A^T
\]

**SVD ↔ PCA**

\[
Z = U\Sigma V^T
\quad\Longrightarrow\quad
\lambda_i = \frac{\sigma_i^2}{n-1},\qquad
\text{scores } = ZV_k
\]

**Least-squares link**

Centred material columns as \(A\), centred wage as \(b\):

\[
\hat{x} = \arg\min\|b-Ax\| \quad\Longleftrightarrow\quad A^TA\hat{x}=A^Tb
\]

(The residual is orthogonal to the column space of the materials.)

---

## Scope & caveats

- The material basket is an **index composite**, not a finished budget or a bid price.  
- Correlation and PCA describe **association**, never causation.  
- Eigenvector signs are arbitrary; interpret relative sizes and signs of the loadings.  
- All numerical examples in the accompanying slides that illustrate Gaussian elimination / LU are purely didactic.

---

## Licence

MIT License – see [LICENSE](LICENSE).

---

*Questions welcome.*  
Sanjey · Shabrish · Sridatta · Rohan · Rohit
