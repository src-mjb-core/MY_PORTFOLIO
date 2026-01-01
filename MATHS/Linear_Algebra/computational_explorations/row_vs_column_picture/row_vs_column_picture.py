import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration for "Advanced" Look ---
plt.style.use('dark_background') # High contrast for vectors
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
}
plt.rcParams.update(params)

def parse_equation_input(prompt_text, expected_vars):
    """
    Parses a string of comma or space separated numbers.
    Example Input: "1 2 3" or "1, 2, 3" -> Returns [1.0, 2.0, 3.0]
    """
    while True:
        user_in = input(prompt_text)
        # Replace commas with spaces and split
        cleaned = user_in.replace(',', ' ').split()
        try:
            values = [float(x) for x in cleaned]
            if len(values) != expected_vars + 1:
                print(f"Error: Expected {expected_vars} coefficients + 1 constant (Total {expected_vars + 1} numbers). Got {len(values)}.")
                continue
            return values[:-1], values[-1]
        except ValueError:
            print("Error: Please enter valid numbers (e.g., '1.5 2 -3').")

def get_user_input():
    print("\n" + "="*60)
    print("   VISUALIZING LINEAR ALGEBRA: Row vs Column    ")
    print("="*60)
    
    # 1. Number of Equations
    while True:
        try:
            num_eq = int(input("\nHow many equations? (1, 2, or 3): "))
            if num_eq in [1, 2, 3]: break
            print("Please enter 1, 2, or 3.")
        except ValueError: continue

    num_vars = 3 if num_eq == 3 else 2
    vars_str = "x, y" if num_vars == 2 else "x, y, z"
    
    print(f"\n--- Input Format ---")
    print(f"Enter coefficients and constant in one line.")
    print(f"Example for 2x + 3y = 5 -> Type: 2 3 5")
    
    coeff_matrix = []
    const_vector = []

    for i in range(num_eq):
        coeffs, constant = parse_equation_input(f"Equation {i+1} ({vars_str}, rhs): ", num_vars)
        coeff_matrix.append(coeffs)
        const_vector.append(constant)

    print("\n--- Visualization Mode ---")
    print("1. Row Picture (Intersections of Lines/Planes)")
    print("2. Column Picture (Vector Addition/Combinations)")
    
    mode = 'row'
    while True:
        c = input("Select (1 or 2): ")
        if c == '1': 
            mode = 'row'; break
        elif c == '2': 
            mode = 'column'; break

    return np.array(coeff_matrix), np.array(const_vector), num_eq, mode

def plot_row_picture_2d(A, b):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set plot limits dynamically based on the constants
    limit = max(10, np.max(np.abs(b)) * 2)
    x_vals = np.linspace(-limit, limit, 400)
    
    colors = ['#FF5555', '#5555FF', '#55FF55'] # Red, Blue, Green (bright for dark bg)

    print("\n> Rendering 2D Row Picture...")
    print("  Visualizing equations as lines. Solution is the intersection point.")

    for i in range(len(b)):
        a_coef, b_coef = A[i]
        rhs = b[i]
        color = colors[i % len(colors)]
        
        label_txt = f"{a_coef}x + {b_coef}y = {rhs}"

        if abs(b_coef) > 1e-6:
            y_vals = (rhs - a_coef * x_vals) / b_coef
            line, = ax.plot(x_vals, y_vals, color=color, linewidth=2, label=label_txt)
            
            # Annotation to identify line
            mid_idx = len(x_vals) // (i + 2) # Offset labels slightly so they don't overlap
            ax.annotate(f"Eq {i+1}", xy=(x_vals[mid_idx], y_vals[mid_idx]), 
                        xytext=(10, 10), textcoords='offset points', 
                        color=color, fontweight='bold', arrowprops=dict(arrowstyle="->", color=color))
        else:
            # Vertical line
            x_int = rhs / a_coef
            ax.axvline(x=x_int, color=color, linewidth=2, label=label_txt)

    # Intersection handling
    if len(b) >= 2:
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            # Verify if it's a true intersection (residuals close to 0)
            if np.allclose(np.dot(A, sol), b):
                ax.plot(sol[0], sol[1], 'o', color='white', markeredgecolor='yellow', markersize=12, zorder=5)
                ax.annotate(f" Solution\n ({sol[0]:.2f}, {sol[1]:.2f})", 
                            xy=(sol[0], sol[1]), xytext=(20, -20), 
                            textcoords='offset points', color='yellow',
                            arrowprops=dict(arrowstyle="->", color='yellow'))
        except:
            pass

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(0, color='white', linewidth=1)
    ax.set_title("Row Picture: 2D Lines Intersection", pad=20)
    ax.legend(loc='upper right')
    plt.show()

def plot_column_picture_2d(A, b):
    rows, cols = A.shape
    
    # Handle the "scalar" case (1 equation) by padding for 2D plot
    if rows == 1:
        A_calc = np.vstack([A, [0, 0]]) # Dummy y-row
        b_calc = np.array([b[0], 0])
    else:
        A_calc = A
        b_calc = b

    # Solve for weights (x, y)
    try:
        sol = np.linalg.lstsq(A_calc, b_calc, rcond=None)[0]
        w1, w2 = sol[0], sol[1]
    except:
        w1, w2 = 0, 0 # Fallback

    fig, ax = plt.subplots(figsize=(10, 8))
    
    print("\n> Rendering 2D Column Picture...")
    print(f"  Visualizing: {w1:.2f} * [Col1] + {w2:.2f} * [Col2] = [b]")

    col1 = A_calc[:, 0]
    col2 = A_calc[:, 1]
    
    origin = np.array([0, 0])

    # 1. Draw "Ghost" Parallelogram (Grid distortion hint)
    # This helps visualize the plane spanned by the columns
    for i in np.linspace(-5, 5, 11):
        # Grid lines parallel to col1
        start = col2 * i * 5 # scale out
        end = col1 * 10 + start # arbitrary length
        # (This is a simplified visual hint, not a full change of basis grid to avoid clutter)

    # 2. Draw Vectors
    def draw_vec(vec, start, color, lbl, style='-', alpha=1.0, width=0.015):
        ax.quiver(start[0], start[1], vec[0], vec[1], 
                  angles='xy', scale_units='xy', scale=1, 
                  color=color, label=lbl, alpha=alpha, width=width)

    # Base Columns (ghosted)
    draw_vec(col1, origin, 'cyan', "Col 1 (Base)", alpha=0.4, width=0.005)
    draw_vec(col2, origin, 'magenta', "Col 2 (Base)", alpha=0.4, width=0.005)

    # Scaled Path (Tip-to-Tail)
    # Step 1: Move along Col 1
    v1_scaled = col1 * w1
    draw_vec(v1_scaled, origin, '#5555FF', f"x*Col1 ({w1:.2f})")
    
    # Step 2: Move along Col 2 (starting from tip of Col 1)
    v2_scaled = col2 * w2
    draw_vec(v2_scaled, v1_scaled, '#FF5555', f"y*Col2 ({w2:.2f})")

    # Dashed line showing the path clearly
    path_x = [0, v1_scaled[0], v1_scaled[0] + v2_scaled[0]]
    path_y = [0, v1_scaled[1], v1_scaled[1] + v2_scaled[1]]
    ax.plot(path_x, path_y, '--', color='white', alpha=0.5)

    # Result Vector b
    draw_vec(b_calc, origin, '#55FF55', "Vector b (Result)", width=0.02)

    # Formatting
    limit = max(np.max(np.abs(b_calc)), np.max(np.abs(v1_scaled + v2_scaled))) * 1.5
    limit = max(limit, 5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(0, color='white', linewidth=1)
    ax.set_title("Column Picture: Tip-to-Tail Vector Addition", pad=20)
    ax.legend()
    plt.show()

def plot_row_picture_3d(A, b):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fix 3D aspect ratio so cubes look cubic
    ax.set_box_aspect([1,1,1])

    limit = 5
    grid_range = np.linspace(-limit, limit, 10)
    X, Y = np.meshgrid(grid_range, grid_range)
    
    colors = ['#00FFFF', '#FF00FF', '#FFFF00'] # CMY colors

    print("\n> Rendering 3D Row Picture...")
    print("  Visualizing equations as Planes.")

    for i in range(len(b)):
        a, b_coef, c = A[i]
        d = b[i]
        
        # Avoid division by zero
        if abs(c) > 1e-6:
            Z = (d - a*X - b_coef*Y) / c
            # Plot surface with manual alpha for transparency
            surf = ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[i], rstride=100, cstride=100)
            # Add wireframe to make the plane orientation clearer
            ax.plot_wireframe(X, Y, Z, alpha=0.5, color=colors[i], rcount=5, ccount=5)
            
            # Labeling planes in legend is tricky in 3D, we use a proxy artist later or print
            print(f"  Plane {i+1}: {a}x + {b_coef}y + {c}z = {d} ({colors[i]})")

    # Intersection Point
    try:
        sol = np.linalg.solve(A, b)
        ax.scatter(sol[0], sol[1], sol[2], color='white', s=200, edgecolor='black', label='Intersection')
        print(f"  Intersection found at: {sol}")
    except:
        print("  No unique intersection point.")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Row Picture: Intersection of Planes")
    
    # Interactive hint
    ax.text2D(0.05, 0.95, "Rotate with Mouse", transform=ax.transAxes, color='white')
    plt.show()

def plot_column_picture_3d(A, b):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    print("\n> Rendering 3D Column Picture...")

    # Calculate weights
    try:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    except:
        w = [0,0,0]

    # Vectors
    origin = np.array([0, 0, 0])
    v1 = A[:, 0] * w[0]
    v2 = A[:, 1] * w[1]
    v3 = A[:, 2] * w[2] if A.shape[1] > 2 else np.array([0,0,0])

    # Path Calculation
    p0 = origin
    p1 = p0 + v1
    p2 = p1 + v2
    p3 = p2 + v3

    # Plotting Helper
    def plot_arrow(start, vec, color, label):
        ax.quiver(start[0], start[1], start[2],
                  vec[0], vec[1], vec[2],
                  color=color, arrow_length_ratio=0.1, linewidth=2, label=label)

    # 1. The Path (Tip-to-Tail)
    plot_arrow(p0, v1, '#5555FF', f"x*Col1 ({w[0]:.2f})")
    plot_arrow(p1, v2, '#FF5555', f"y*Col2 ({w[1]:.2f})")
    if A.shape[1] > 2:
        plot_arrow(p2, v3, '#FFFF55', f"z*Col3 ({w[2]:.2f})")

    # 2. The Result (b)
    plot_arrow(origin, b, '#55FF55', "Vector b")

    # 3. Dashed Guidelines (The "Ghost" Path)
    # Drawing lines connecting the tips to emphasize the addition structure
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    zs = [p0[2], p1[2], p2[2], p3[2]]
    ax.plot(xs, ys, zs, 'w--', alpha=0.6, linewidth=1, label="Combination Path")

    # Axis Limits
    limit = max(np.max(np.abs(b)), 5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Column Picture: 3D Vector Path")
    ax.legend()
    plt.show()

def main():
    # 1. Enhanced Input
    A, b, num_eq, mode = get_user_input()
    
    # 2. Routing
    if num_eq == 3:
        if mode == 'row': plot_row_picture_3d(A, b)
        else: plot_column_picture_3d(A, b)
    else:
        if mode == 'row': plot_row_picture_2d(A, b)
        else: plot_column_picture_2d(A, b)

if __name__ == "__main__":
    main()
