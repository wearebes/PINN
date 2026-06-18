# Level-Set Static Bubble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated `levelset_static_bubble` benchmark that tests whether the NN27 level-set curvature closure can maintain the stationary circular-bubble surface-tension balance.

**Architecture:** Keep this benchmark self-contained under `levelset_static_bubble/`. Stage 1 is a frozen level-set geometry and curvature-diagnostic pipeline; Stage 2 is split into force-assembly sanity checks, a primary steady Stokes solve for `Ca_eq(N)`, and a limited implicit transient solve for representative `Ca_max(t)` curves.

**Tech Stack:** Python, NumPy, SciPy sparse solvers, pandas/matplotlib for outputs, pytest for contracts, and the existing checkpoint reader in `evaluate.shared` only as a read-only model-loading dependency.

---

## 0. Requirements And Target Files

### 0.1 Hard Requirements

This plan deliberately replaces the earlier `cfd_static_bubble` interpretation. The following are hard requirements:

- Do not mix VOF into this benchmark.
- Do not mix height-function baselines into this benchmark.
- Do not use the cell-spacing convention `h = 1 / N`.
- Use level-set signed distance only:

  \[
  \phi(\mathbf x)=\|\mathbf x-\mathbf x_c\|_2-R.
  \]

- Use only these core diagnostics:

  \[
  \kappa_{\mathrm{std}},\qquad Ca_{\max}(t),
  \]

  plus the resolution summary:

  \[
  Ca_{\mathrm{eq}}(N).
  \]

- Treat the level set as frozen:

  \[
  \phi(\mathbf x,t)=\phi_0(\mathbf x).
  \]

- All implementation files must live under:

  ```text
  levelset_static_bubble/
  ```

- Existing project files may be imported or read, but must not be modified by this benchmark plan.

- Include an `EXACT` curvature method:

  \[
  \kappa_{ij}^{EXACT}=\frac{1}{R}=2.5.
  \]

  This is the solver/discretization floor for `Ca_eq` and `Ca_max(t)`. Any NN result must be interpreted relative to this floor.

### 0.2 Target Files For The Implementation

Create these files during implementation:

```text
levelset_static_bubble/IMPLEMENTATION_PLAN.md
levelset_static_bubble/__init__.py
levelset_static_bubble/contracts.py
levelset_static_bubble/geometry.py
levelset_static_bubble/interface_nodes.py
levelset_static_bubble/features.py
levelset_static_bubble/d4.py
levelset_static_bubble/curvature.py
levelset_static_bubble/model_adapter.py
levelset_static_bubble/stokes.py
levelset_static_bubble/metrics.py
levelset_static_bubble/io.py
levelset_static_bubble/run_matrix.py
levelset_static_bubble/plot_results.py
levelset_static_bubble/tests/test_contracts.py
levelset_static_bubble/tests/test_geometry.py
levelset_static_bubble/tests/test_interface_nodes.py
levelset_static_bubble/tests/test_features.py
levelset_static_bubble/tests/test_curvature.py
levelset_static_bubble/tests/test_d4.py
levelset_static_bubble/tests/test_stokes.py
levelset_static_bubble/results/.gitkeep
```

The implementation may read these existing repo interfaces, without editing them:

```text
evaluate/shared.py
model/model.py
out/*/*_hgradient.pt
out/*/*_hgradient.csv
```

The NN adapter must reject 9D checkpoints. Accepted checkpoints must satisfy:

\[
\text{raw\_feature\_dim}=27
\]

and the feature order must be either:

\[
\texttt{phi9+nx9+ny9}
\]

or a transform whose raw input is:

\[
\texttt{pca18(phi9+nx9+ny9)}.
\]

The adapter metadata must write these fields into every summary row:

```text
raw_feature_dim,transform_source,feature_order,target_scale,checkpoint_hash,feature_order_assumed
```

If the checkpoint metadata does not explicitly store the stencil order, use:

```text
feature_order_assumed=phi9_top_to_bottom_training_order
```

and the implementation must still use the exact order in Section 1.4.

---

## 1. Mathematical Contract

Every implementation step must be checked against this section. If code behavior conflicts with any equation here, the code is wrong.

### 1.1 Resolutions And Grid Spacing

\[
N\in\{64,128,256,512\}
\]

Here `N` means the number of grid points per unit physical length, not the number of cells.

\[
\boxed{h_N=\frac{1}{N-1}}
\]

For the quadrant domain:

\[
\Omega_Q=[0,1]\times[0,1],
\qquad
x_i=i h_N,\quad y_j=j h_N,
\qquad
i,j=0,\dots,N-1.
\]

Therefore:

\[
\boxed{\text{grid}_Q(N)=N\times N}
\]

For the full-circle domain:

\[
\Omega_F=[0,2]\times[0,2],
\qquad
x_i=i h_N,\quad y_j=j h_N,
\qquad
i,j=0,\dots,2N-2.
\]

Therefore:

\[
\boxed{\text{grid}_F(N)=(2N-1)\times(2N-1)}
\]

The fixed radius is:

\[
\boxed{R=0.4}
\]

The reference ratios are:

| `N` | `h_N` | `R/h_N` |
|---:|---:|---:|
| 64 | `1/63` | `25.2` |
| 128 | `1/127` | `50.8` |
| 256 | `1/255` | `102.0` |
| 512 | `1/511` | `204.4` |

### 1.2 Geometries

Configuration Q:

\[
\Omega_Q=[0,1]^2,\qquad \mathbf x_c^Q=(0,0),
\]

\[
\boxed{\phi_Q(x,y)=\sqrt{x^2+y^2}-R}
\]

\[
\Gamma_Q=\{(x,y)\in\Omega_Q:\phi_Q(x,y)=0\}.
\]

Configuration F:

\[
\Omega_F=[0,2]^2,\qquad \mathbf x_c^F=(1,1),
\]

\[
\boxed{\phi_F(x,y)=\sqrt{(x-1)^2+(y-1)^2}-R}
\]

\[
\Gamma_F=\{(x,y)\in\Omega_F:\phi_F(x,y)=0\}.
\]

The exact circle curvature is identical in both configurations:

\[
\boxed{\kappa_{\mathrm{exact}}=\frac{1}{R}=2.5}
\]

### 1.3 Interface Node Set

The curvature is computed only on selected cross/interface nodes:

\[
\boxed{
\mathcal I_h=
\{(i,j):|\phi_{ij}|\le 1.5h_N,\ C_{ij}=1,\ S_{ij}\ \mathrm{valid}\}
}
\]

The cross-node condition is:

\[
\boxed{
C_{ij}=1
\iff
\exists(a,b)\in\{(1,0),(-1,0),(0,1),(0,-1)\}
\ \mathrm{s.t.}\ 
\phi_{ij}\phi_{i+a,j+b}\le 0
}
\]

Stencil validity means every value needed by the 3x3 stencil and by the central-difference normals exists:

\[
S_{ij}\ \mathrm{valid}
\iff
\phi_{i+a,j+b}\ \mathrm{and}\ \phi_{i+a\pm1,j+b},\phi_{i+a,j+b\pm1}
\ \mathrm{exist}
\quad
\forall a,b\in\{-1,0,1\}.
\]

For Configuration Q, boundary ghost values must use analytic symmetry:

\[
\phi_Q(-x,y)=\phi_Q(x,y),\qquad \phi_Q(x,-y)=\phi_Q(x,y).
\]

### 1.4 Feature Construction

For every selected node:

\[
(i,j)\in\mathcal I_h,
\]

the 3x3 level-set stencil is:

\[
\phi_{9,ij}=\{\phi_{i+a,j+b}:a,b\in\{-1,0,1\}\}.
\]

The flattened order must match the existing training order:

\[
(-1,1),(0,1),(1,1),(-1,0),(0,0),(1,0),(-1,-1),(0,-1),(1,-1).
\]

At each stencil point:

\[
D_x\phi_{i+a,j+b}=
\frac{\phi_{i+a+1,j+b}-\phi_{i+a-1,j+b}}{2h_N},
\]

\[
D_y\phi_{i+a,j+b}=
\frac{\phi_{i+a,j+b+1}-\phi_{i+a,j+b-1}}{2h_N}.
\]

With:

\[
\eta=10^{-14},
\]

the normal is:

\[
\boxed{
n_x=\frac{D_x\phi}{\sqrt{(D_x\phi)^2+(D_y\phi)^2+\eta}}
}
\]

\[
\boxed{
n_y=\frac{D_y\phi}{\sqrt{(D_x\phi)^2+(D_y\phi)^2+\eta}}
}
\]

The NN27 raw feature is:

\[
\boxed{
raw27_{ij}=
\left[
\frac{\phi_{9,ij}}{h_N},
\ n_{x,9,ij},
\ n_{y,9,ij}
\right]
}
\]

### 1.5 Curvature Methods

The benchmark methods are exactly:

\[
\boxed{EXACT,\quad LS\text{-}FD,\quad NN27\_RAW,\quad NN27\_D4}
\]

The exact baseline is:

\[
\boxed{
\kappa_{ij}^{EXACT}=\frac{1}{R}=2.5
}
\]

This method is mandatory because it measures the level-set CSF and Stokes solver floor:

\[
\boxed{
Ca_{\mathrm{floor}}(N)=Ca_{\mathrm{eq}}^{EXACT}(N)
}
\]

Other methods must be interpreted through both the absolute value and the floor-relative values:

\[
\boxed{
Ca_{\mathrm{excess}}^{method}(N)=
Ca_{\mathrm{eq}}^{method}(N)-Ca_{\mathrm{floor}}(N)
}
\]

\[
\boxed{
Ca_{\mathrm{rel}}^{method}(N)=
\frac{Ca_{\mathrm{eq}}^{method}(N)}
{Ca_{\mathrm{floor}}(N)+10^{-30}}
}
\]

For the finite-difference level-set baseline, define:

\[
\phi_x=\frac{1}{2}(\phi_{i+1,j}-\phi_{i-1,j}),
\quad
\phi_y=\frac{1}{2}(\phi_{i,j+1}-\phi_{i,j-1}),
\]

\[
\phi_{xx}=\phi_{i+1,j}-2\phi_{i,j}+\phi_{i-1,j},
\quad
\phi_{yy}=\phi_{i,j+1}-2\phi_{i,j}+\phi_{i,j-1},
\]

\[
\phi_{xy}=\frac{1}{4}
(\phi_{i+1,j+1}-\phi_{i+1,j-1}-\phi_{i-1,j+1}+\phi_{i-1,j-1}).
\]

Then:

\[
\boxed{
\widehat y^{FD}_{ij}
=
\frac{
\phi_{xx}\phi_y^2-2\phi_x\phi_y\phi_{xy}+\phi_{yy}\phi_x^2
}{
(\phi_x^2+\phi_y^2)^{3/2}
}
\approx h_N\kappa_{ij}
}
\]

and:

\[
\boxed{\kappa^{FD}_{ij}=\frac{\widehat y^{FD}_{ij}}{h_N}}
\]

For NN27:

\[
\boxed{
\widehat y_{ij}=MLP(raw27_{ij})\approx h_N\kappa_{ij}
}
\]

\[
\boxed{
\kappa_{ij}^{NN}=\frac{\widehat y_{ij}}{h_N}
}
\]

For D4 averaging, use the eight orthogonal maps:

\[
D_4=\left\{
\begin{bmatrix}1&0\\0&1\end{bmatrix},
\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\begin{bmatrix}-1&0\\0&-1\end{bmatrix},
\begin{bmatrix}0&1\\-1&0\end{bmatrix},
\begin{bmatrix}1&0\\0&-1\end{bmatrix},
\begin{bmatrix}-1&0\\0&1\end{bmatrix},
\begin{bmatrix}0&1\\1&0\end{bmatrix},
\begin{bmatrix}0&-1\\-1&0\end{bmatrix}
\right\}.
\]

Each transform must move both scalar stencil positions and vector normals:

\[
\phi_9(a,b)\mapsto \phi_9(g(a,b)),
\qquad
\mathbf n(a,b)\mapsto g\,\mathbf n(g(a,b)).
\]

Then:

\[
\boxed{
\kappa_{ij}^{NN-D4}
=
\frac{1}{8h_N}
\sum_{g\in D_4}MLP(g\cdot raw27_{ij})
}
\]

### 1.6 Curvature Statistics

Let:

\[
N_{\mathcal I}=|\mathcal I_h|.
\]

Then:

\[
\bar\kappa_h=
\frac{1}{N_{\mathcal I}}
\sum_{(i,j)\in\mathcal I_h}\kappa_{ij},
\]

\[
\boxed{
\kappa_{\mathrm{std}}=
\left[
\frac{1}{N_{\mathcal I}}
\sum_{(i,j)\in\mathcal I_h}
(\kappa_{ij}-\bar\kappa_h)^2
\right]^{1/2}
}
\]

For the exact continuous circle:

\[
\kappa_{ij}=2.5
\quad\Longrightarrow\quad
\kappa_{\mathrm{std}}=0.
\]

The discrete implementation must report the measured value; it must not silently replace measured values with the continuous value.

### 1.7 Level-Set Surface Force

Use the level-set CSF force:

\[
\boxed{
\mathbf f_\sigma=
\sigma\kappa_h\delta_\epsilon(\phi)\nabla_h\phi
}
\]

with:

\[
\boxed{\epsilon=1.5h_N}
\]

and:

\[
\boxed{
\delta_\epsilon(\phi)=
\begin{cases}
\dfrac{1}{2\epsilon}
\left[
1+\cos\left(\dfrac{\pi\phi}{\epsilon}\right)
\right],
&|\phi|\le\epsilon,\\
0,
&|\phi|>\epsilon.
\end{cases}
}
\]

The curvature is computed on the cross/interface set:

\[
\mathcal I_h.
\]

The force is not applied only on `\mathcal I_h`. Define the Eulerian narrow band:

\[
\boxed{
\mathcal B_h=
\{(i,j):|\phi_{ij}|\le\epsilon\}
}
\]

For every band node, extend curvature from the nearest selected interface node:

\[
\boxed{
m(i,j)=
\arg\min_{(m,n)\in\mathcal I_h}
\|\mathbf x_{ij}-\mathbf x_{mn}\|_2
}
\]

\[
\boxed{
\kappa^{ext}_{ij}=\kappa_{m(i,j)}
}
\]

Then apply force on the full band:

\[
\boxed{
\mathbf f_{\sigma,ij}
=
\sigma\kappa^{ext}_{ij}
\delta_\epsilon(\phi_{ij})
\nabla_h\phi_{ij},
\qquad
(i,j)\in\mathcal B_h
}
\]

Outside the band:

\[
\boxed{\mathbf f_{\sigma,ij}=0.}
\]

The curvature statistics remain on `\mathcal I_h` only:

\[
\boxed{
\kappa_{\mathrm{std}}
=
\operatorname{std}\{\kappa_{ij}:(i,j)\in\mathcal I_h\}.
}
\]

### 1.8 Stokes Response And Capillary Metrics

Stage 2 uses steady Stokes as the primary solver for `Ca_eq`:

\[
\boxed{
0=-\nabla p+\mu\Delta\mathbf u+\mathbf f_\sigma
}
\]

\[
\boxed{\nabla\cdot\mathbf u=0}
\]

The steady result is:

\[
\boxed{\mathbf u_\infty=\mathrm{SteadyStokes}(\mathbf f_\sigma)}
\]

and the equilibrium capillary number is:

\[
\boxed{
Ca_{\mathrm{eq}}(N)=
\frac{\mu}{\sigma}
\max_{i,j}\|\mathbf u_{\infty,ij}\|_2
}
\]

Only representative cases use transient response for plotting `Ca_max(t*)`. The transient equation is:

\[
\boxed{
\rho_f\frac{\partial\mathbf u}{\partial t}
=
-\nabla p+\mu\Delta\mathbf u+\mathbf f_\sigma
}
\]

\[
\boxed{\nabla\cdot\mathbf u=0}
\]

with:

\[
\mathbf u(\mathbf x,0)=0.
\]

Transient stepping must be implicit Euler:

\[
\boxed{
\frac{\rho_f}{\Delta t}(\mathbf u^{n+1}-\mathbf u^n)
=
-\nabla p^{n+1}
+\mu\Delta_h\mathbf u^{n+1}
+\mathbf f_\sigma
}
\]

\[
\boxed{\nabla_h\cdot\mathbf u^{n+1}=0.}
\]

Do not split the transient step with an explicit diffusive CFL bound. Use:

\[
\boxed{
\Delta t = \Delta t^\ast_{\mathrm{record}}t_\sigma
}
\]

for implicit transient runs.

The Stokes discretization must use a MAC staggered grid:

| Variable | Location |
|---|---|
| `p` | scalar nodes/cell centers |
| `u` | x-faces |
| `v` | y-faces |

If an implementation intentionally uses a collocated grid instead, it must be marked as a deviation and must pass:

\[
\boxed{
\|\nabla_h\cdot\mathbf u\|_\infty < 10^{-8}
}
\]

after projection or steady solve.

Use:

\[
\rho_f=1,\qquad \sigma=1,\qquad R=0.4,\qquad D=2R=0.8,\qquad La=12000.
\]

The dynamic viscosity is:

\[
\boxed{
\mu=\sqrt{\frac{\rho_f\sigma D}{La}}
=\sqrt{\frac{0.8}{12000}}
=8.16496580927726\times10^{-3}
}
\]

The capillary time is:

\[
\boxed{
t_\sigma=\sqrt{\frac{\rho_fD^3}{\sigma}}
=\sqrt{0.512}
=0.7155417527999327
}
\]

Use:

\[
t^\ast=\frac{t}{t_\sigma},
\qquad
t^\ast_{\max}=200,
\qquad
\Delta t^\ast_{\mathrm{record}}=0.25.
\]

The maximum velocity is:

\[
\boxed{
U_{\max}(t)=
\max_{i,j}\sqrt{u_{ij}(t)^2+v_{ij}(t)^2}
}
\]

The maximum capillary number is:

\[
\boxed{
Ca_{\max}(t)=\frac{\mu U_{\max}(t)}{\sigma}
}
\]

For optional transient diagnostics, the late-time window is:

\[
\boxed{
\mathcal T_{\mathrm{eq}}=
\{t^\ast_k:t^\ast_k\ge0.6t^\ast_{\max}\}
=
\{t^\ast_k:t^\ast_k\ge120\}
}
\]

The transient late-time average is a diagnostic only:

\[
\boxed{
Ca_{\mathrm{late}}(N)=
\frac{1}{|\mathcal T_{\mathrm{eq}}|}
\sum_{t^\ast_k\in\mathcal T_{\mathrm{eq}}}
Ca_{\max}(t^\ast_k;N)
}
\]

The reported full-resolution `Ca_eq(N)` must come from the steady Stokes solve unless the implementation explicitly marks a run as `solver=transient`.

---

## 2. Stage 1: Frozen Level-Set Geometry And Curvature Diagnostics

Stage 1 must produce `kappa_mean(N)`, `kappa_std(N)`, and `kappa_linf_error(N)` for each configuration, resolution, and method. It must not solve flow.

### Task 1.1: Implement Contracts And Geometry

**Files:**

- Create: `levelset_static_bubble/contracts.py`
- Create: `levelset_static_bubble/geometry.py`
- Test: `levelset_static_bubble/tests/test_contracts.py`
- Test: `levelset_static_bubble/tests/test_geometry.py`

- [ ] **Step 1: Add immutable constants**

  `contracts.py` must define:

  ```python
  RESOLUTIONS = (64, 128, 256, 512)
  RADIUS = 0.4
  RHO_F = 1.0
  SIGMA = 1.0
  LAPLACE = 12000.0
  DIAMETER = 2.0 * RADIUS
  MU = (RHO_F * SIGMA * DIAMETER / LAPLACE) ** 0.5
  T_SIGMA = (RHO_F * DIAMETER**3 / SIGMA) ** 0.5
  ETA_NORMAL = 1.0e-14
  BAND_FACTOR = 1.5
  METHODS = ("EXACT", "LS-FD", "NN27_RAW", "NN27_D4")
  CONFIGS = ("Q", "F")
  ```

- [ ] **Step 2: Add grid helpers**

  `geometry.py` must implement:

  ```python
  def h_from_N(N: int) -> float:
      if N not in RESOLUTIONS:
          raise ValueError(f"N must be one of {RESOLUTIONS}, got {N}.")
      return 1.0 / float(N - 1)

  def grid_size(config: str, N: int) -> int:
      h_from_N(N)
      if config == "Q":
          return N
      if config == "F":
          return 2 * N - 1
      raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")
  ```

- [ ] **Step 3: Add coordinate and phi builders**

  `geometry.py` must implement:

  ```python
  def coordinates(config: str, N: int) -> tuple[np.ndarray, np.ndarray]:
      h = h_from_N(N)
      n = grid_size(config, N)
      axis = np.arange(n, dtype=np.float64) * h
      return np.meshgrid(axis, axis, indexing="ij")

  def phi(config: str, N: int) -> np.ndarray:
      x, y = coordinates(config, N)
      if config == "Q":
          return np.sqrt(x * x + y * y) - RADIUS
      if config == "F":
          return np.sqrt((x - 1.0) ** 2 + (y - 1.0) ** 2) - RADIUS
      raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")
  ```

- [ ] **Step 4: Write contract tests**

  `test_contracts.py` must assert:

  ```python
  assert h_from_N(64) == 1.0 / 63.0
  assert grid_size("Q", 64) == 64
  assert grid_size("F", 64) == 127
  assert abs(MU - 8.16496580927726e-3) < 1e-15
  assert abs(T_SIGMA - 0.7155417527999327) < 1e-15
  ```

- [ ] **Step 5: Write geometry tests**

  `test_geometry.py` must assert:

  ```python
  q = phi("Q", 64)
  assert q.shape == (64, 64)
  assert abs(q[0, 0] + RADIUS) < 1e-15
  assert abs(q[25, 0]) < 1.0 / 63.0

  f = phi("F", 64)
  assert f.shape == (127, 127)
  assert abs(f[63, 63] + RADIUS) < 1e-15
  assert abs(f[63 + 25, 63]) < 1.0 / 63.0
  ```

- [ ] **Step 6: Run tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_contracts.py levelset_static_bubble/tests/test_geometry.py -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 1.2: Implement Interface Node Selection And Analytic Ghost Access

**Files:**

- Create: `levelset_static_bubble/interface_nodes.py`
- Modify: `levelset_static_bubble/geometry.py`
- Test: `levelset_static_bubble/tests/test_interface_nodes.py`

- [ ] **Step 1: Add analytic phi value access**

  `geometry.py` must expose:

  ```python
  def phi_value(config: str, x: float, y: float) -> float:
      if config == "Q":
          return float((x * x + y * y) ** 0.5 - RADIUS)
      if config == "F":
          return float(((x - 1.0) ** 2 + (y - 1.0) ** 2) ** 0.5 - RADIUS)
      raise ValueError(f"config must be 'Q' or 'F', got {config!r}.")

  def phi_at_index(config: str, N: int, i: int, j: int) -> float:
      h = h_from_N(N)
      if config == "Q":
          return phi_value("Q", abs(i * h), abs(j * h))
      n = grid_size("F", N)
      if not (0 <= i < n and 0 <= j < n):
          raise IndexError(f"F index out of bounds for grid {n}: {(i, j)}")
      return phi_value("F", i * h, j * h)
  ```

- [ ] **Step 2: Implement cross-node selection**

  `interface_nodes.py` must use the exact condition:

  ```python
  FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

  def is_cross_node(config: str, N: int, i: int, j: int) -> bool:
      center = phi_at_index(config, N, i, j)
      for di, dj in FOUR_NEIGHBORS:
          try:
              neighbor = phi_at_index(config, N, i + di, j + dj)
          except IndexError:
              continue
          if center * neighbor <= 0.0:
              return True
      return False
  ```

- [ ] **Step 3: Implement stencil validity**

  `interface_nodes.py` must require the 5x5 support implied by 3x3 normals:

  ```python
  def has_valid_stencil(config: str, N: int, i: int, j: int) -> bool:
      if config == "Q":
          return True
      n = grid_size("F", N)
      return 2 <= i <= n - 3 and 2 <= j <= n - 3
  ```

  This is valid because Configuration Q uses analytic symmetry ghosts, while Configuration F has no ghost rule and the circle is far from the boundary.

- [ ] **Step 4: Implement selected node array**

  ```python
  def interface_indices(config: str, N: int) -> np.ndarray:
      h = h_from_N(N)
      n = grid_size(config, N)
      out: list[tuple[int, int]] = []
      for i in range(n):
          for j in range(n):
              if abs(phi_at_index(config, N, i, j)) <= BAND_FACTOR * h:
                  if is_cross_node(config, N, i, j) and has_valid_stencil(config, N, i, j):
                      out.append((i, j))
      return np.asarray(out, dtype=np.int64)
  ```

- [ ] **Step 5: Write node-selection tests**

  Tests must assert:

  ```python
  for config in ("Q", "F"):
      idx = interface_indices(config, 64)
      assert idx.ndim == 2 and idx.shape[1] == 2
      assert len(idx) > 0
      h = h_from_N(64)
      for i, j in idx:
          assert abs(phi_at_index(config, 64, int(i), int(j))) <= 1.5 * h + 1e-15
          assert is_cross_node(config, 64, int(i), int(j))
          assert has_valid_stencil(config, 64, int(i), int(j))
  ```

- [ ] **Step 6: Run tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_interface_nodes.py -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 1.3: Implement Raw27 Features And D4 Transformations

**Files:**

- Create: `levelset_static_bubble/features.py`
- Create: `levelset_static_bubble/d4.py`
- Test: `levelset_static_bubble/tests/test_features.py`
- Test: `levelset_static_bubble/tests/test_d4.py`

- [ ] **Step 1: Implement the exact 3x3 order**

  `features.py` must define:

  ```python
  PHI9_OFFSETS = (
      (-1, 1), (0, 1), (1, 1),
      (-1, 0), (0, 0), (1, 0),
      (-1, -1), (0, -1), (1, -1),
  )
  ```

- [ ] **Step 2: Implement stencil extraction**

  ```python
  def phi9(config: str, N: int, i: int, j: int) -> np.ndarray:
      return np.asarray(
          [phi_at_index(config, N, i + di, j + dj) for di, dj in PHI9_OFFSETS],
          dtype=np.float32,
      )
  ```

- [ ] **Step 3: Implement normal extraction**

  ```python
  def normal_at(config: str, N: int, i: int, j: int) -> tuple[float, float]:
      h = h_from_N(N)
      dx = (phi_at_index(config, N, i + 1, j) - phi_at_index(config, N, i - 1, j)) / (2.0 * h)
      dy = (phi_at_index(config, N, i, j + 1) - phi_at_index(config, N, i, j - 1)) / (2.0 * h)
      denom = (dx * dx + dy * dy + ETA_NORMAL) ** 0.5
      return float(dx / denom), float(dy / denom)

  def normal9(config: str, N: int, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
      vals = [normal_at(config, N, i + di, j + dj) for di, dj in PHI9_OFFSETS]
      nx = np.asarray([v[0] for v in vals], dtype=np.float32)
      ny = np.asarray([v[1] for v in vals], dtype=np.float32)
      return nx, ny
  ```

- [ ] **Step 4: Implement raw27**

  ```python
  def raw27(config: str, N: int, i: int, j: int) -> np.ndarray:
      h = h_from_N(N)
      p9 = phi9(config, N, i, j) / np.float32(h)
      nx, ny = normal9(config, N, i, j)
      return np.concatenate([p9, nx, ny]).astype(np.float32, copy=False)
  ```

- [ ] **Step 5: Implement D4 maps**

  `d4.py` must implement the eight matrices in Section 1.5 and transform `raw27` by:

  ```python
  def transform_raw27(raw: np.ndarray, matrix: np.ndarray) -> np.ndarray:
      p9 = raw[:9]
      nx = raw[9:18]
      ny = raw[18:27]
      lookup = {offset: k for k, offset in enumerate(PHI9_OFFSETS)}
      out_p = np.empty(9, dtype=np.float32)
      out_nx = np.empty(9, dtype=np.float32)
      out_ny = np.empty(9, dtype=np.float32)
      for k, offset in enumerate(PHI9_OFFSETS):
          src_offset = tuple(np.asarray(matrix).T @ np.asarray(offset, dtype=np.int64))
          src = lookup[src_offset]
          vec = np.asarray([nx[src], ny[src]], dtype=np.float32)
          transformed_vec = np.asarray(matrix, dtype=np.float32) @ vec
          out_p[k] = p9[src]
          out_nx[k] = transformed_vec[0]
          out_ny[k] = transformed_vec[1]
      return np.concatenate([out_p, out_nx, out_ny]).astype(np.float32, copy=False)
  ```

- [ ] **Step 6: Write feature tests**

  Tests must assert:

  ```python
  idx = interface_indices("F", 64)
  i, j = map(int, idx[len(idx) // 2])
  r = raw27("F", 64, i, j)
  assert r.shape == (27,)
  assert np.isfinite(r).all()
  nx = r[9:18]
  ny = r[18:27]
  assert np.max(np.abs(np.sqrt(nx * nx + ny * ny) - 1.0)) < 1e-6
  ```

- [ ] **Step 7: Write D4 tests**

  Tests must assert:

  ```python
  idx = interface_indices("F", 64)
  i, j = map(int, idx[len(idx) // 2])
  r = raw27("F", 64, i, j)
  transformed = [transform_raw27(r, m) for m in D4_MATRICES]
  assert len(transformed) == 8
  assert all(x.shape == (27,) for x in transformed)
  assert all(np.isfinite(x).all() for x in transformed)
  identity = transform_raw27(r, D4_MATRICES[0])
  np.testing.assert_allclose(identity, r, rtol=0.0, atol=0.0)

  for matrix in D4_MATRICES:
      recovered = transform_raw27(transform_raw27(r, matrix), matrix.T)
      np.testing.assert_allclose(recovered, r, rtol=0.0, atol=1e-6)

  for value in transformed:
      nx = value[9:18]
      ny = value[18:27]
      np.testing.assert_allclose(np.sqrt(nx * nx + ny * ny), 1.0, rtol=0.0, atol=1e-6)
  ```

- [ ] **Step 8: Run tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_features.py levelset_static_bubble/tests/test_d4.py -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 1.4: Implement Curvature Methods And Stage-1 Outputs

**Files:**

- Create: `levelset_static_bubble/curvature.py`
- Create: `levelset_static_bubble/model_adapter.py`
- Create: `levelset_static_bubble/metrics.py`
- Create: `levelset_static_bubble/io.py`
- Create: `levelset_static_bubble/run_matrix.py`
- Test: `levelset_static_bubble/tests/test_curvature.py`

- [ ] **Step 1: Implement `EXACT` curvature**

  `curvature.py` must expose:

  ```python
  def curvature_exact(config: str, N: int, i: int, j: int) -> float:
      return 1.0 / RADIUS
  ```

  The arguments are kept so all methods share one call signature.

- [ ] **Step 2: Implement `LS-FD` curvature**

  `curvature.py` must implement the formula from Section 1.5 and return physical curvature:

  ```python
  def curvature_lsf_fd(config: str, N: int, i: int, j: int) -> float:
      h = h_from_N(N)
      p = {offset: phi_at_index(config, N, i + offset[0], j + offset[1]) for offset in PHI9_OFFSETS}
      px = 0.5 * (p[(1, 0)] - p[(-1, 0)])
      py = 0.5 * (p[(0, 1)] - p[(0, -1)])
      pxx = p[(1, 0)] - 2.0 * p[(0, 0)] + p[(-1, 0)]
      pyy = p[(0, 1)] - 2.0 * p[(0, 0)] + p[(0, -1)]
      pxy = 0.25 * (p[(1, 1)] - p[(1, -1)] - p[(-1, 1)] + p[(-1, -1)])
      denom = (px * px + py * py) ** 1.5
      if denom <= 0.0 or not np.isfinite(denom):
          raise ValueError(f"non-finite curvature denominator at {(config, N, i, j)}")
      hkappa = (pxx * py * py - 2.0 * px * py * pxy + pyy * px * px) / denom
      return float(hkappa / h)
  ```

- [ ] **Step 3: Implement model adapter**

  `model_adapter.py` must import only:

  ```python
  from evaluate.shared import load_model_from_checkpoint, resolve_feature_transform, predict_hkappa_full_batch
  ```

  It must validate:

  ```python
  transform["raw_feature_dim"] == 27
  ```

  and reject incompatible checkpoints with a `ValueError`.

  It must expose metadata:

  ```python
  {
      "raw_feature_dim": 27,
      "transform_source": transform_source,
      "feature_order": transform.get("feature_order", ""),
      "target_scale": "h*kappa",
      "checkpoint_hash": sha256_file(checkpoint_path),
      "feature_order_assumed": "phi9_top_to_bottom_training_order",
  }
  ```

- [ ] **Step 4: Implement `NN27_RAW` and `NN27_D4`**

  `curvature.py` must expose:

  ```python
  def curvature_nn27_raw(model_bundle: ModelBundle, raw: np.ndarray, h: float) -> float:
      y = model_bundle.predict_hkappa(raw.reshape(1, 27))[0]
      return float(y / h)

  def curvature_nn27_d4(model_bundle: ModelBundle, raw: np.ndarray, h: float) -> float:
      transformed = np.stack([transform_raw27(raw, matrix) for matrix in D4_MATRICES], axis=0)
      y = model_bundle.predict_hkappa(transformed)
      return float(np.mean(y) / h)
  ```

- [ ] **Step 5: Implement curvature statistics**

  `metrics.py` must expose:

  ```python
  def curvature_stats(values: np.ndarray) -> dict[str, float]:
      arr = np.asarray(values, dtype=np.float64).reshape(-1)
      if arr.size == 0:
          raise ValueError("curvature_stats requires at least one selected interface node")
      mean = float(np.mean(arr))
      return {
          "kappa_exact": 1.0 / RADIUS,
          "kappa_mean": mean,
          "kappa_std": float(np.sqrt(np.mean((arr - mean) ** 2))),
          "kappa_linf_error": float(np.max(np.abs(arr - 1.0 / RADIUS))),
      }
  ```

- [ ] **Step 6: Implement stage-1 run command**

  `run_matrix.py` must support:

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage curvature \
    --configs Q,F \
    --methods EXACT,LS-FD,NN27_RAW,NN27_D4 \
    --resolutions 64,128,256,512 \
    --checkpoint out/256/baseline_256_hgradient.pt \
    --out levelset_static_bubble/results
  ```

  Output:

  ```text
  levelset_static_bubble/results/curvature_nodes.csv
  levelset_static_bubble/results/curvature_summary.csv
  ```

- [ ] **Step 7: Write curvature tests**

  Tests must assert `EXACT` is exactly constant and `LS-FD` is finite on selected nodes:

  ```python
  idx = interface_indices("F", 64)
  exact = np.asarray([curvature_exact("F", 64, int(i), int(j)) for i, j in idx], dtype=np.float64)
  np.testing.assert_allclose(exact, 2.5, rtol=0.0, atol=0.0)

  values = np.asarray([curvature_lsf_fd("F", 64, int(i), int(j)) for i, j in idx], dtype=np.float64)
  assert np.isfinite(values).all()
  stats = curvature_stats(values)
  assert stats["kappa_exact"] == 2.5
  assert stats["kappa_std"] >= 0.0
  assert stats["kappa_linf_error"] >= 0.0
  ```

  Tests must also assert the `LS-FD` mean approaches the exact value under refinement:

  ```python
  idx64 = interface_indices("F", 64)
  idx128 = interface_indices("F", 128)
  mean64 = np.mean([curvature_lsf_fd("F", 64, int(i), int(j)) for i, j in idx64])
  mean128 = np.mean([curvature_lsf_fd("F", 128, int(i), int(j)) for i, j in idx128])
  assert abs(mean128 - 2.5) < abs(mean64 - 2.5)
  ```

- [ ] **Step 8: Run Stage-1 tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_curvature.py -q
  ```

  Expected result:

  ```text
  passed
  ```

- [ ] **Step 9: Run Stage-1 matrix**

  Start with `EXACT` and `LS-FD` to verify the non-NN path:

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage curvature \
    --configs Q,F \
    --methods EXACT,LS-FD \
    --resolutions 64,128,256,512 \
    --out levelset_static_bubble/results
  ```

  Then run NN methods with an explicit 27D checkpoint:

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage curvature \
    --configs Q,F \
    --methods NN27_RAW,NN27_D4 \
    --resolutions 64,128,256,512 \
    --checkpoint out/256/baseline_256_hgradient.pt \
    --out levelset_static_bubble/results
  ```

- [ ] **Step 10: Commit Stage 1**

  ```bash
  git add levelset_static_bubble
  git commit -m "feat: add level-set static bubble curvature diagnostics"
  ```

---

## 3. Stage 2: Force Sanity, Steady Stokes, And Limited Transient Reporting

Stage 2 consumes Stage-1 curvature fields. It must run in this order:

\[
\boxed{\text{2A force assembly sanity}\rightarrow\text{2B steady Stokes}\rightarrow\text{2C representative transient}}
\]

The full matrix uses steady Stokes for `Ca_eq`. Transient `Ca_max(t*)` is only for representative plots.

### Task 2.1: Implement Level-Set CSF Force On The Whole Band

**Files:**

- Create: `levelset_static_bubble/stokes.py`
- Modify: `levelset_static_bubble/metrics.py`
- Test: `levelset_static_bubble/tests/test_stokes.py`

- [ ] **Step 1: Implement regularized delta**

  ```python
  def delta_epsilon(phi_value: np.ndarray, epsilon: float) -> np.ndarray:
      values = np.asarray(phi_value, dtype=np.float64)
      out = np.zeros_like(values, dtype=np.float64)
      mask = np.abs(values) <= epsilon
      out[mask] = 0.5 / epsilon * (1.0 + np.cos(np.pi * values[mask] / epsilon))
      return out
  ```

- [ ] **Step 2: Implement band nodes**

  ```python
  def band_indices(config: str, N: int) -> np.ndarray:
      h = h_from_N(N)
      n = grid_size(config, N)
      out: list[tuple[int, int]] = []
      for i in range(n):
          for j in range(n):
              if abs(phi_at_index(config, N, i, j)) <= BAND_FACTOR * h:
                  out.append((i, j))
      return np.asarray(out, dtype=np.int64)
  ```

- [ ] **Step 3: Implement nearest-interface-node curvature extension**

  ```python
  def node_xy(N: int, indices: np.ndarray) -> np.ndarray:
      h = h_from_N(N)
      idx = np.asarray(indices, dtype=np.float64)
      return idx * h

  def extend_kappa_to_band(
      N: int,
      interface_idx: np.ndarray,
      interface_kappa: np.ndarray,
      band_idx: np.ndarray,
  ) -> np.ndarray:
      from scipy.spatial import cKDTree

      interface_idx = np.asarray(interface_idx, dtype=np.int64)
      band_idx = np.asarray(band_idx, dtype=np.int64)
      kappa = np.asarray(interface_kappa, dtype=np.float64).reshape(-1)
      if interface_idx.shape[0] != kappa.shape[0]:
          raise ValueError("interface_idx and interface_kappa lengths differ")
      tree = cKDTree(node_xy(N, interface_idx))
      _, nearest = tree.query(node_xy(N, band_idx), k=1)
      return kappa[np.asarray(nearest, dtype=np.int64)]
  ```

- [ ] **Step 4: Implement force assembly on `B_h`**

  ```python
  def surface_force(
      config: str,
      N: int,
      interface_idx: np.ndarray,
      interface_kappa: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      n = grid_size(config, N)
      h = h_from_N(N)
      epsilon = BAND_FACTOR * h
      bidx = band_indices(config, N)
      bkappa = extend_kappa_to_band(N, interface_idx, interface_kappa, bidx)
      fx = np.zeros((n, n), dtype=np.float64)
      fy = np.zeros((n, n), dtype=np.float64)
      for (i, j), kij in zip(bidx.astype(np.int64), bkappa):
          ph = phi_at_index(config, N, int(i), int(j))
          nx, ny = normal_at(config, N, int(i), int(j))
          weight = SIGMA * float(kij) * float(delta_epsilon(np.asarray([ph]), epsilon)[0])
          fx[int(i), int(j)] = weight * nx
          fy[int(i), int(j)] = weight * ny
      return fx, fy, bidx
  ```

- [ ] **Step 5: Implement force-balance diagnostics**

  ```python
  def force_balance(config: str, N: int, fx: np.ndarray, fy: np.ndarray) -> dict[str, float]:
      h = h_from_N(N)
      total_x = float(np.sum(np.asarray(fx, dtype=np.float64)) * h * h)
      total_y = float(np.sum(np.asarray(fy, dtype=np.float64)) * h * h)
      return {
          "force_sum_x": total_x,
          "force_sum_y": total_y,
          "force_sum_norm": float((total_x * total_x + total_y * total_y) ** 0.5),
      }
  ```

- [ ] **Step 6: Write force tests**

  ```python
  idx = interface_indices("F", 64)
  kappa = np.full(len(idx), 2.5, dtype=np.float64)
  fx, fy, bidx = surface_force("F", 64, idx, kappa)
  assert fx.shape == (127, 127)
  assert fy.shape == (127, 127)
  assert len(bidx) >= len(idx)
  assert np.isfinite(fx).all()
  assert np.isfinite(fy).all()
  assert np.count_nonzero(fx) > 0
  assert np.count_nonzero(fy) > 0

  balance = force_balance("F", 64, fx, fy)
  assert balance["force_sum_norm"] < 1e-8
  ```

- [ ] **Step 7: Run force tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_stokes.py::test_surface_force -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 2.2: Implement Steady MAC Stokes Solver

**Files:**

- Modify: `levelset_static_bubble/stokes.py`
- Modify: `levelset_static_bubble/metrics.py`
- Test: `levelset_static_bubble/tests/test_stokes.py`

- [ ] **Step 1: Implement MAC force interpolation**

  The scalar force arrays `fx, fy` live on the level-set scalar grid. Interpolate them to MAC faces:

  \[
  f^u_{i+1/2,j}=\frac{1}{2}(f^x_{i,j}+f^x_{i+1,j}),
  \]

  \[
  f^v_{i,j+1/2}=\frac{1}{2}(f^y_{i,j}+f^y_{i,j+1}).
  \]

  Boundary normal velocity faces are fixed to zero.

- [ ] **Step 2: Implement MAC operators**

  Use pressure/scalars on shape `(n, n)`, x-velocity on `(n + 1, n)`, and y-velocity on `(n, n + 1)`.

  Divergence:

  \[
  (\nabla_h\cdot\mathbf u)_{ij}
  =
  \frac{u_{i+1/2,j}-u_{i-1/2,j}}{h}
  +
  \frac{v_{i,j+1/2}-v_{i,j-1/2}}{h}.
  \]

  Pressure gradient:

  \[
  (G_xp)_{i+1/2,j}=\frac{p_{i+1,j}-p_{i,j}}{h},
  \qquad
  (G_yp)_{i,j+1/2}=\frac{p_{i,j+1}-p_{i,j}}{h}.
  \]

  Velocity Laplacians use free-slip boundary closure:

  \[
  \mathbf u\cdot\mathbf n_b=0,\qquad
  \frac{\partial \mathbf u_t}{\partial n_b}=0.
  \]

- [ ] **Step 3: Implement steady saddle-point solve**

  Assemble:

  \[
  \boxed{
  \begin{bmatrix}
  \mu L_u & 0 & -G_x\\
  0 & \mu L_v & -G_y\\
  D_x & D_y & 0
  \end{bmatrix}
  \begin{bmatrix}
  u_\infty\\
  v_\infty\\
  p
  \end{bmatrix}
  =
  \begin{bmatrix}
  -f^u\\
  -f^v\\
  0
  \end{bmatrix}
  }
  \]

  Fix the pressure gauge with:

  \[
  p_{0,0}=0.
  \]

  `stokes.py` must expose:

  ```python
  def solve_steady_stokes_mac(config: str, N: int, fx: np.ndarray, fy: np.ndarray) -> dict[str, np.ndarray | float]:
      fu, fv = interpolate_force_to_mac(config, N, fx, fy)
      matrix, rhs, slices = assemble_steady_mac_system(config, N, fu, fv)
      solution = scipy.sparse.linalg.spsolve(matrix, rhs)
      u, v, p = unpack_mac_solution(solution, slices)
      div_linf = divergence_linf(config, N, u, v)
      u_node, v_node = mac_velocity_to_scalar_nodes(u, v)
      U_max = umax(u_node, v_node)
      return {
          "u": u,
          "v": v,
          "p": p,
          "div_linf": div_linf,
          "U_max": U_max,
          "Ca_eq": ca_from_umax(U_max),
      }
  ```

  The helper functions in this snippet must be implemented in `stokes.py` as part of this step; they are not optional external dependencies.

  Return keys:

  ```text
  u,v,p,div_linf,U_max,Ca_eq
  ```

- [ ] **Step 4: Implement metrics**

  ```python
  def umax(u: np.ndarray, v: np.ndarray) -> float:
      return float(max(np.max(np.abs(u)), np.max(np.abs(v))))

  def ca_from_umax(U_max: float) -> float:
      return MU * float(U_max) / SIGMA
  ```

- [ ] **Step 5: Write steady Stokes tests**

  ```python
  n = grid_size("F", 64)
  fx = np.zeros((n, n), dtype=np.float64)
  fy = np.zeros((n, n), dtype=np.float64)
  result = solve_steady_stokes_mac("F", 64, fx, fy)
  assert result["U_max"] == 0.0
  assert result["Ca_eq"] == 0.0
  assert result["div_linf"] < 1e-12
  ```

  For exact circular force:

  ```python
  idx = interface_indices("F", 64)
  kappa = np.full(len(idx), 2.5, dtype=np.float64)
  fx, fy, _ = surface_force("F", 64, idx, kappa)
  result = solve_steady_stokes_mac("F", 64, fx, fy)
  assert np.isfinite(result["Ca_eq"])
  assert result["Ca_eq"] >= 0.0
  assert result["div_linf"] < 1e-8
  ```

- [ ] **Step 6: Run steady Stokes tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_stokes.py::test_steady_stokes -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 2.3: Implement Optional Implicit Transient For Representative Curves

**Files:**

- Modify: `levelset_static_bubble/stokes.py`
- Modify: `levelset_static_bubble/metrics.py`
- Test: `levelset_static_bubble/tests/test_stokes.py`

- [ ] **Step 1: Implement transient step size without diffusive substeps**

  ```python
  def transient_dt(record_dt_star: float = 0.25) -> float:
      if record_dt_star <= 0.0:
          raise ValueError(f"record_dt_star must be positive, got {record_dt_star}")
      return float(record_dt_star) * T_SIGMA
  ```

- [ ] **Step 2: Implement implicit transient solve**

  Each step solves:

  \[
  \boxed{
  \begin{bmatrix}
  \rho_f/\Delta t-\mu L_u & 0 & G_x\\
  0 & \rho_f/\Delta t-\mu L_v & G_y\\
  D_x & D_y & 0
  \end{bmatrix}
  \begin{bmatrix}
  u^{n+1}\\
  v^{n+1}\\
  p^{n+1}
  \end{bmatrix}
  =
  \begin{bmatrix}
  \rho_f u^n/\Delta t + f^u\\
  \rho_f v^n/\Delta t + f^v\\
  0
  \end{bmatrix}
  }
  \]

  `stokes.py` must expose:

  ```python
  def solve_transient_stokes_mac(
      config: str,
      N: int,
      fx: np.ndarray,
      fy: np.ndarray,
      *,
      t_star_max: float,
      record_dt_star: float = 0.25,
  ) -> dict[str, np.ndarray]:
      dt = transient_dt(record_dt_star)
      step_count = int(np.ceil(float(t_star_max) / float(record_dt_star)))
      fu, fv = interpolate_force_to_mac(config, N, fx, fy)
      u, v = zero_mac_velocity(config, N)
      rows: list[dict[str, float]] = []
      for step in range(step_count + 1):
          t_star = float(step) * float(record_dt_star)
          u_node, v_node = mac_velocity_to_scalar_nodes(u, v)
          U_max = umax(u_node, v_node)
          rows.append({
              "t": t_star * T_SIGMA,
              "t_star": t_star,
              "U_max": U_max,
              "Ca_max": ca_from_umax(U_max),
              "div_linf": divergence_linf(config, N, u, v),
          })
          if step == step_count:
              break
          matrix, rhs, slices = assemble_transient_mac_system(config, N, fu, fv, u, v, dt)
          solution = scipy.sparse.linalg.spsolve(matrix, rhs)
          u, v, _p = unpack_mac_solution(solution, slices)
      return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]}
  ```

  The helper functions in this snippet must be implemented in `stokes.py` as part of this step; they must reuse the same MAC operators as the steady solver.

  Return keys:

  ```text
  t,t_star,U_max,Ca_max,div_linf
  ```

- [ ] **Step 3: Implement transient late-time diagnostic**

  ```python
  def ca_late(t_star: np.ndarray, ca: np.ndarray, t_star_max: float) -> float:
      t = np.asarray(t_star, dtype=np.float64)
      values = np.asarray(ca, dtype=np.float64)
      mask = t >= 0.6 * float(t_star_max)
      if not np.any(mask):
          raise ValueError("ca_late requires at least one sample in the late-time window")
      return float(np.mean(values[mask]))
  ```

- [ ] **Step 4: Write transient tests**

  ```python
  n = grid_size("F", 64)
  fx = np.zeros((n, n), dtype=np.float64)
  fy = np.zeros((n, n), dtype=np.float64)
  result = solve_transient_stokes_mac("F", 64, fx, fy, t_star_max=0.5, record_dt_star=0.25)
  np.testing.assert_allclose(result["Ca_max"], 0.0, rtol=0.0, atol=0.0)
  assert np.max(result["div_linf"]) < 1e-12
  ```

- [ ] **Step 5: Run transient tests**

  ```bash
  python -m pytest levelset_static_bubble/tests/test_stokes.py::test_transient_stokes -q
  ```

  Expected result:

  ```text
  passed
  ```

### Task 2.4: Run Stage-2 Matrices And Reporting

**Files:**

- Modify: `levelset_static_bubble/run_matrix.py`
- Modify: `levelset_static_bubble/io.py`
- Modify: `levelset_static_bubble/plot_results.py`

- [ ] **Step 1: Add Stage-2 commands**

  `run_matrix.py` must support:

  ```text
  --stage force
  --stage steady
  --stage transient
  ```

- [ ] **Step 2: Write output schemas**

  `io.py` must write:

  ```text
  levelset_static_bubble/results/force_summary.csv
  levelset_static_bubble/results/steady_summary.csv
  levelset_static_bubble/results/timeseries.csv
  levelset_static_bubble/results/summary.csv
  ```

  `force_summary.csv` columns:

  ```text
  config,method,N,grid_n,h,R_over_h,n_interface,n_band,force_sum_x,force_sum_y,force_sum_norm
  ```

  `steady_summary.csv` and `summary.csv` columns:

  ```text
  config,method,N,grid_n,h,R_over_h,n_interface,n_band,kappa_exact,kappa_mean,kappa_std,kappa_linf_error,force_sum_norm,Ca_eq,Ca_floor,Ca_excess,Ca_rel,div_linf,solver,checkpoint_path,raw_feature_dim,transform_source,feature_order,target_scale,checkpoint_hash,feature_order_assumed
  ```

  `timeseries.csv` columns:

  ```text
  config,method,N,grid_n,h,R_over_h,t,t_star,U_max,Ca_max,div_linf,solver
  ```

- [ ] **Step 3: Run force sanity matrix first**

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage force \
    --configs Q,F \
    --methods EXACT,LS-FD,NN27_RAW,NN27_D4 \
    --resolutions 64,128,256,512 \
    --checkpoint out/256/baseline_256_hgradient.pt \
    --out levelset_static_bubble/results/force
  ```

- [ ] **Step 4: Run steady full matrix**

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage steady \
    --configs Q,F \
    --methods EXACT,LS-FD,NN27_RAW,NN27_D4 \
    --resolutions 64,128,256,512 \
    --checkpoint out/256/baseline_256_hgradient.pt \
    --out levelset_static_bubble/results/steady
  ```

  `Ca_floor`, `Ca_excess`, and `Ca_rel` must be computed by grouping rows by:

  ```text
  config,N
  ```

  and using the `EXACT` row as the floor.

- [ ] **Step 5: Run representative transient only**

  ```bash
  python -m levelset_static_bubble.run_matrix \
    --stage transient \
    --configs Q,F \
    --methods EXACT,LS-FD,NN27_RAW,NN27_D4 \
    --resolutions 128 \
    --checkpoint out/256/baseline_256_hgradient.pt \
    --t-star-max 200 \
    --record-dt-star 0.25 \
    --out levelset_static_bubble/results/transient_N128
  ```

  Do not run all resolutions through transient until the steady results and representative transient curves have been inspected.

- [ ] **Step 6: Plot final figures**

  ```bash
  python -m levelset_static_bubble.plot_results \
    --steady levelset_static_bubble/results/steady \
    --transient levelset_static_bubble/results/transient_N128 \
    --out levelset_static_bubble/results/figures
  ```

  Required figures:

  ```text
  kappa_std_vs_N_Q.png
  kappa_std_vs_N_F.png
  ca_eq_vs_N_Q.png
  ca_eq_vs_N_F.png
  ca_excess_vs_N_Q.png
  ca_excess_vs_N_F.png
  ca_max_tstar_Q_N128.png
  ca_max_tstar_F_N128.png
  ```

- [ ] **Step 7: Commit Stage 2**

  ```bash
  git add levelset_static_bubble
  git commit -m "feat: add level-set static bubble force and Stokes diagnostics"
  ```

---

## 4. Acceptance Gates

### 4.1 Mathematical Gates

A run is invalid if any of these are false:

\[
h_N=\frac{1}{N-1}
\]

\[
\text{grid}_Q(N)=N\times N
\]

\[
\text{grid}_F(N)=(2N-1)\times(2N-1)
\]

\[
\kappa_{\mathrm{exact}}=2.5
\]

\[
\mathrm{methods}=\{EXACT,LS\text{-}FD,NN27\_RAW,NN27\_D4\}
\]

\[
raw27=[\phi_9/h_N,n_{x,9},n_{y,9}]
\]

\[
\mathcal I_h\ \text{computes curvature},\qquad
\mathcal B_h=\{(i,j):|\phi_{ij}|\le1.5h_N\}\ \text{receives force}
\]

\[
\phi(\mathbf x,t)=\phi_0(\mathbf x)
\]

\[
Ca_{\mathrm{eq}}(N)=
\frac{\mu}{\sigma}\max_{ij}\|\mathbf u_{\infty,ij}\|_2
\quad
\text{from steady Stokes}
\]

\[
Ca_{\max}(t)=\frac{\mu}{\sigma}\max_{ij}\sqrt{u_{ij}^2+v_{ij}^2}
\]

\[
\|\nabla_h\cdot\mathbf u\|_\infty<10^{-8}
\quad
\text{for nonzero-force Stokes solves}
\]

### 4.2 Scope Gates

A pull request or local change is invalid if it modifies files outside `levelset_static_bubble/`, except for changes explicitly approved before implementation.

### 4.3 Reporting Gates

Final reporting must include:

- `curvature_summary.csv`
- `force_summary.csv`
- `steady_summary.csv`
- `timeseries.csv`
- `summary.csv`
- `kappa_mean`, `kappa_std`, and `kappa_linf_error`
- `Ca_floor`, `Ca_excess`, and `Ca_rel`
- `kappa_std` vs `N` for Q and F
- `Ca_eq` vs `N` for Q and F
- `Ca_excess` vs `N` for Q and F
- `Ca_max(t*)` for a representative `N`, default `N=128`

No report may claim that the discrete benchmark should produce exactly:

\[
Ca_{\max}(t)=0
\]

The exact zero statement is the continuous theoretical target. The implementation reports measured discrete values.

---

## 5. Self-Review Checklist

- The plan is isolated under `levelset_static_bubble/`.
- The plan excludes VOF and height functions.
- The plan uses `h_N=1/(N-1)` everywhere.
- The quadrant grid is `N x N`.
- The full-circle grid is `(2N-1) x (2N-1)`.
- The method set includes `EXACT`.
- The interface node set is mathematically defined before feature construction.
- The band set `B_h` is mathematically defined before force construction.
- The NN27 feature is exactly `[phi9/h_N, nx9, ny9]`.
- D4 tests include inverse consistency and normal norm preservation.
- Checkpoint metadata includes raw dimension, feature order or assumed feature order, target scale, transform source, and checkpoint hash.
- The finite-difference baseline is level-set finite difference, not height function.
- The level set is frozen during the Stokes response.
- The output metrics keep `kappa_mean` even though the headline curvature plot uses `kappa_std`.
- The full matrix uses steady Stokes for `Ca_eq(N)`.
- The transient solver has no explicit diffusive substep restriction.
- The plan separates Stage 1 curvature diagnostics from Stage 2 force sanity, steady Stokes, and representative transient response.
