import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
plt.style.use('dark_background')
params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.titlesize': 16
}
plt.rcParams.update(params)

def parse_equation_input(prompt_text, expected_vars):
    while True:
        user_in = input(prompt_text)
        cleaned = user_in.replace(',', ' ').split()
        try:
            values = [float(x) for x in cleaned]
            if len(values) != expected_vars + 1:
                print(f"Error: Expected {expected_vars} coeffs + 1 constant. Got {len(values)}.")
                continue
            return values[:-1], values[-1]
        except ValueError:
            print("Error: Please enter valid numbers.")

def get_user_input():
    print("\n" + "="*70)
    print("   LINEAR ALGEBRA DUAL-VIEW: Row & Column Pictures (1D - 6D)   ")
    print("="*70)
    
    while True:
        try:
            dim = int(input("\nEnter dimension (number of variables 1-6): "))
            if 1 <= dim <= 6: break
            print("Please keep it between 1 and 6 for visualization.")
        except ValueError: continue

    print(f"\n--- Strategy for {dim} Dimensions ---")
    if dim == 2:
        print("Standard 2D Geometry (Lines & Flat Vectors).")
    elif dim >= 3:
        print(f"3D Projection Mode:")
        print("  - Row Picture: Showing 3D slice of the intersection.")
        print("  - Column Picture: Showing 3D shadow of {dim}D vectors.")
    
    # Square system: Num Equations = Num Dimensions
    num_eq = dim 
    coeff_matrix = []
    const_vector = []
    
    print(f"\n--- Input Equations ---")
    vars_labels = ["x1", "x2", "x3", "x4", "x5", "x6"]
    
    for i in range(num_eq):
        prompt = f"Eq {i+1} ({', '.join(vars_labels[:dim])}, rhs): "
        coeffs, constant = parse_equation_input(prompt, dim)
        coeff_matrix.append(coeffs)
        const_vector.append(constant)

    return np.array(coeff_matrix), np.array(const_vector), dim

# ==========================================
#          2D DRAWING LOGIC
# ==========================================

def draw_row_2d(ax, A, b):
    limit = max(10, np.max(np.abs(b)) * 2)
    x = np.linspace(-limit, limit, 100)
    colors = ['#FF5555', '#5555FF']

    # Draw Lines
    for i in range(len(b)):
        a_coef, b_coef = A[i]
        rhs = b[i]
        color = colors[i % 2]
        
        if abs(b_coef) > 1e-6:
            y = (rhs - a_coef*x) / b_coef
            ax.plot(x, y, color=color, linewidth=2, label=f"Eq {i+1}")
        else:
            ax.axvline(rhs/a_coef, color=color, linewidth=2, label=f"Eq {i+1}")

    # Draw Intersection
    try:
        sol = np.linalg.solve(A, b)
        ax.plot(sol[0], sol[1], 'wo', markeredgecolor='yellow', markersize=10, label="Solution")
        ax.text(sol[0], sol[1]+1, f"({sol[0]:.1f}, {sol[1]:.1f})", color='yellow', ha='center')
    except: pass
    
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_title("ROW PICTURE\n(Intersection of Lines)", color='yellow', pad=10)
    ax.axhline(0, color='white', lw=1); ax.axvline(0, color='white', lw=1)
    ax.legend()

def draw_col_2d(ax, A, b):
    origin = np.array([0, 0])
    try: weights = np.linalg.lstsq(A, b, rcond=None)[0]
    except: weights = [0, 0]
    
    # 1. Draw "Ghost" Vectors (Original Basis)
    ax.quiver(*origin, *A[:,0], color='cyan', scale=1, scale_units='xy', alpha=0.3, width=0.005)
    ax.quiver(*origin, *A[:,1], color='magenta', scale=1, scale_units='xy', alpha=0.3, width=0.005)

    # 2. Draw Scaled Path (Tip-to-Tail)
    v1 = A[:,0] * weights[0]
    v2 = A[:,1] * weights[1]
    
    # Step 1
    ax.quiver(*origin, *v1, color='#5555FF', scale=1, scale_units='xy', label=f"x*Col1")
    # Step 2
    ax.quiver(*v1, *v2, color='#FF5555', scale=1, scale_units='xy', label=f"y*Col2")
    
    # Ghost Path Line
    ax.plot([0, v1[0], v1[0]+v2[0]], [0, v1[1], v1[1]+v2[1]], 'w--', alpha=0.5)

    # 3. Result
    ax.quiver(*origin, *b, color='#55FF55', scale=1, scale_units='xy', label="Vector b", width=0.015)
    
    limit = max(5, np.max(np.abs(b)) * 1.5)
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit)
    ax.set_title("COLUMN PICTURE\n(Combination of Vectors)", color='lime', pad=10)
    ax.axhline(0, color='white', lw=1); ax.axvline(0, color='white', lw=1)
    ax.legend()

# ==========================================
#       3D/HIGH-DIM DRAWING LOGIC
# ==========================================

def get_projection_matrix(dim):
    # Squash N-dims to 3D for visualization
    if dim <= 3: return np.eye(dim)
    P = np.zeros((3, dim))
    P[0:3, 0:3] = np.eye(3)
    extra_dirs = [[0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, -0.5, 0.2]]
    for i in range(3, dim): P[:, i] = extra_dirs[i-3]
    return P

def draw_row_high_dim(ax, A, b, dim):
    x = np.linspace(-5, 5, 10)
    X, Y = np.meshgrid(x, x)
    colors = plt.cm.hsv(np.linspace(0, 1, len(b) + 1))

    # Slice View: Plot planes assuming x4...xN = 0
    for i in range(len(b)):
        coeffs = A[i]
        rhs = b[i]
        # Use first 3 coords for the "slice"
        a, coeff_b, c = coeffs[0], coeffs[1], coeffs[2] if dim >= 3 else 0
        
        if abs(c) > 1e-6:
            Z = (rhs - a*X - coeff_b*Y) / c
            ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[i], rstride=10, cstride=10)
        
    try:
        true_sol = np.linalg.solve(A, b)
        vis_sol = true_sol[0:3] # Project to 3D
        ax.scatter(vis_sol[0], vis_sol[1], vis_sol[2], color='white', s=100, edgecolor='black', label='Solution (Proj)')
    except: pass

    ax.set_title(f"ROW PICTURE\n({dim}D Hyperplane Slice)", color='yellow')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

def draw_col_high_dim(ax, A, b, dim):
    try: weights = np.linalg.solve(A, b)
    except: weights = np.zeros(dim)

    P = get_projection_matrix(dim)
    
    # Project columns and b to 3D
    proj_cols = [P @ A[:, i] for i in range(dim)]
    proj_b = P @ b
    
    origin = np.array([0., 0., 0.])
    curr = origin.copy()
    colors = plt.cm.rainbow(np.linspace(0, 1, dim))

    # Tip-to-Tail Path
    for i in range(dim):
        vec_scaled = proj_cols[i] * weights[i]
        ax.quiver(curr[0], curr[1], curr[2], 
                  vec_scaled[0], vec_scaled[1], vec_scaled[2],
                  color=colors[i], arrow_length_ratio=0.1, label=f"x{i}")
        curr += vec_scaled

    # Result b
    ax.quiver(0, 0, 0, proj_b[0], proj_b[1], proj_b[2], color='white', arrow_length_ratio=0.1, label='b')
    
    # Connecting Line
    path_pts = [origin]
    curr = origin.copy()
    for i in range(dim):
        curr = curr + (proj_cols[i] * weights[i])
        path_pts.append(curr.copy())
    path_pts = np.array(path_pts)
    ax.plot(path_pts[:,0], path_pts[:,1], path_pts[:,2], 'w--', alpha=0.5)

    limit = max(5, np.max(np.abs(path_pts)))
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.set_title(f"COLUMN PICTURE\n({dim}D Vector Shadow)", color='lime')

# ==========================================
#              MAIN EXECUTION
# ==========================================

def main():
    A, b, dim = get_user_input()
    
    # Create the Canvas
    if dim == 2:
        # 2D Setup: 1 row, 2 cols
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        draw_row_2d(ax1, A, b)
        draw_col_2d(ax2, A, b)
    else:
        # 3D/High-Dim Setup: 1 row, 2 cols (both 3D projection)
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Consistent viewing angle
        ax1.view_init(elev=20, azim=-45)
        ax2.view_init(elev=20, azim=-45)
        
        draw_row_high_dim(ax1, A, b, dim)
        draw_col_high_dim(ax2, A, b, dim)

    plt.suptitle(f"Linear System Visualization ({dim} Variables)", color='white', fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
