"""
CONSTRUCTION COSTS THROUGH LINEAR ALGEBRA - whole project, numpy only.
Run:  python construction_project.py

IMPORTANT: the monthly data below is RANDOM and ILLUSTRATIVE.
Replace the DATA section with your real, cited data (24+ months, one market).
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def title(t):
    print("\n" + "=" * 10, t, "=" * 10)

# ======================================================
# DATA (slide 2): rows = months, columns = 4 series
# ======================================================
rng = np.random.default_rng(0)
n = 36
cement = 100 + np.cumsum(rng.normal(0, 1.0, n))
steel  = 0.8 * cement + 20 + rng.normal(0, 2.0, n)
lumber = 50 + np.cumsum(rng.normal(0, 1.5, n))
wage   = 20 + 0.05 * np.arange(n) + rng.normal(0, 0.3, n)
X = np.column_stack([cement, steel, lumber, wage])
names = ["cement", "steel", "lumber", "wage"]

# To use real data instead, e.g. from a CSV with 4 columns and a header row:
# X = np.loadtxt("data.csv", delimiter=",", skiprows=1)

# ======================================================
# STEP 2 (slide 3): standardise, covariance, correlation
# ======================================================
title("Standardise and covariance")
Xc = X - X.mean(axis=0)                      # centre
Z = Xc / X.std(axis=0, ddof=1)               # centre and scale
S = Xc.T @ Xc / (n - 1)                      # covariance matrix
R = Z.T @ Z / (n - 1)                        # correlation matrix
print("Covariance S:\n", S)
print("Correlation R:\n", R)

# ======================================================
# SLIDE 4: Gaussian elimination and LU  (A x = b)
# ======================================================
title("LU decomposition (slide 4)")
A = np.array([[2., 1., 1.],
              [4., 3., 3.],
              [8., 7., 9.]])
b = np.array([7., 19., 49.])

def lu(A):
    """A = L U without row swaps (fine for this example)."""
    m = len(A)
    L, U = np.eye(m), A.copy()
    for j in range(m - 1):
        for i in range(j + 1, m):
            L[i, j] = U[i, j] / U[j, j]      # multiplier
            U[i] -= L[i, j] * U[j]           # eliminate
    return L, U

L, U = lu(A)
print("L:\n", L, "\nU:\n", U, "\nL @ U == A:", np.allclose(L @ U, A))

y = np.zeros(3)                              # forward:  L y = b
for i in range(3):
    y[i] = b[i] - L[i, :i] @ y[:i]
x = np.zeros(3)                              # backward: U x = y
for i in range(2, -1, -1):
    x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
print("y =", y, " x =", x, " matches np.linalg.solve:", np.allclose(x, np.linalg.solve(A, b)))

# ======================================================
# SLIDES 5-7: rank, inverse, subspaces, independence
# ======================================================
title("Rank, inverse, subspaces (slides 5-7)")
print("rank(A) =", np.linalg.matrix_rank(A))
print("A inverse:\n", np.linalg.inv(A))
print("rank of centred data Xc =", np.linalg.matrix_rank(Xc), "(4 means the series are independent)")

# Four fundamental subspaces via SVD
Ua, sa, Vta = np.linalg.svd(A)
r = np.sum(sa > 1e-10)
print("dim column space =", r, " dim null space =", A.shape[1] - r, "(rank-nullity: sums to n)")

# ======================================================
# SLIDE 8: basket as a linear transformation  T: R^4 -> R^2
# ======================================================
title("Basket map (slide 8)")
w = np.array([0.5, 0.35, 0.15])              # ASSUMPTION - use documented cost shares
assert np.isclose(w.sum(), 1) and np.all(w >= 0)
A_map = np.array([[*w, 0.],
                  [0., 0., 0., 1.]])
T = X @ A_map.T                              # each row = (q, wage)
print("first 3 months (q, wage):\n", T[:3])
print("rank(A_map) =", np.linalg.matrix_rank(A_map), "-> kernel dim = 4 - 2 =", 4 - np.linalg.matrix_rank(A_map))
print("Cov(Ax) = A S A^T:\n", A_map @ S @ A_map.T)

# ======================================================
# SLIDE 9: dot product, cosine = correlation, Gram-Schmidt / QR
# ======================================================
title("Gram-Schmidt and QR (slide 9)")
print("cos(theta) of centred cement, steel =", round(Xc[:, 0] @ Xc[:, 1] /
      (np.linalg.norm(Xc[:, 0]) * np.linalg.norm(Xc[:, 1])), 3),
      " | correlation =", round(R[0, 1], 3))

def gram_schmidt(M):
    Q = np.zeros_like(M)
    for j in range(M.shape[1]):
        v = M[:, j].copy()
        for i in range(j):
            v -= (M[:, j] @ Q[:, i]) * Q[:, i]   # remove earlier directions
        Q[:, j] = v / np.linalg.norm(v)
    return Q

Q = gram_schmidt(Xc)
print("Q^T Q = I :", np.allclose(Q.T @ Q, np.eye(4)))

# ======================================================
# SLIDE 10: projection and least squares (materials -> wage)
# ======================================================
title("Projection and least squares (slide 10)")
Am, bw = Xc[:, :3], Xc[:, 3]                 # centred materials, centred wage
x_hat = np.linalg.solve(Am.T @ Am, Am.T @ bw)    # normal equations
P = Am @ np.linalg.inv(Am.T @ Am) @ Am.T
p = P @ bw
e = bw - p
print("x_hat =", x_hat, " (np.linalg.lstsq:", np.linalg.lstsq(Am, bw, rcond=None)[0], ")")
print("P^2 = P :", np.allclose(P @ P, P), "| error perpendicular to columns:", np.allclose(Am.T @ e, 0))
print("R^2 =", round(1 - e @ e / (bw @ bw), 3), "(how well materials move with wage - association, not cause)")

# ======================================================
# SLIDE 11: eigenvalues and eigenvectors
# ======================================================
title("Eigenvalues (slide 11)")
lam, V = np.linalg.eigh(R)                   # eigh for symmetric matrices
lam, V = lam[::-1], V[:, ::-1]               # largest first
print("eigenvalues of R:", lam, " sum =", lam.sum())
print("R = Q Lambda Q^T :", np.allclose(R, V @ np.diag(lam) @ V.T))

# ======================================================
# SLIDE 12: positive definite tests
# ======================================================
title("Positive definite tests (slide 12)")
print("all eigenvalues > 0 :", bool(np.all(lam > 0)))
try:
    np.linalg.cholesky(R)
    print("Cholesky works -> positive definite")
except np.linalg.LinAlgError:
    print("Cholesky failed -> NOT positive definite")
print("condition number =", round(lam[0] / lam[-1], 2), "(large means nearly redundant series)")

# ======================================================
# SLIDE 13: SVD
# ======================================================
title("SVD (slide 13)")
Uz, sz, Vtz = np.linalg.svd(Z, full_matrices=False)
print("singular values:", sz)
print("Z = U S V^T :", np.allclose(Z, Uz @ np.diag(sz) @ Vtz))
print("lambda = sigma^2/(n-1):", sz**2 / (n - 1), "(matches eigenvalues of R:", np.allclose(sz**2 / (n - 1), lam), ")")

# ======================================================
# SLIDE 14: PCA
# ======================================================
title("PCA (slide 14)")
Vz = Vtz.T                                   # loadings (columns)
ratio = lam / lam.sum()
print("explained ratio:", ratio, " cumulative:", np.cumsum(ratio))
k = int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)      # keep >= 90% variance
print("keep k =", k, "components")
Y = Z @ Vz[:, :k]                            # scores: months in new axes
print("Loadings (rows = series, columns = PC):")
for nme, row in zip(names, Vz[:, :k]):
    print(f"  {nme:7s}", row)
print("Cov(Y) is diagonal of lambda :", np.allclose(np.cov(Y, rowvar=False), np.diag(lam[:k])))
print("\nNote: signs of loadings are arbitrary; read relative sizes. PCA shows association, not cause.")