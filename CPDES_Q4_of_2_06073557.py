import numpy as np
import matplotlib.pyplot as plt
from CPDES_Q4_06073557_CODE import solve_2d_leapfrog


delta = 1 / 5
xp = 1 / 2
yp = 1
pml_width=2.0

def initial_cond(X, Y):
    """
    The initial pulse is
        u(x,y,0) = cos(pi r / (2 delta))   for r <= delta
                 = 0                       otherwise
    where r = sqrt((x-xp)^2 + (y-yp)^2).
    """
    r = np.sqrt((X - xp) ** 2 + (Y - yp) ** 2)

    U_0 = np.zeros_like(X)
    mask = r <= delta
    U_0[mask] = np.cos(np.pi * r[mask] / (2 * delta))
    return U_0


def dx(A, h):
    """
    Second-order finite difference approximation to A_x.
    """
    D = np.zeros_like(A)
    D[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * h)
    D[0, :] = (-3 * A[0, :] + 4 * A[1, :] - A[2, :]) / (2 * h)
    D[-1, :] = (3 * A[-1, :] - 4 * A[-2, :] + A[-3, :]) / (2 * h)
    return D


def dy(A, h):
    """
    Second-order finite difference approximation to A_y.
    """
    D = np.zeros_like(A)
    D[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2 * h)
    D[:, 0] = (-3 * A[:, 0] + 4 * A[:, 1] - A[:, 2]) / (2 * h)
    D[:, -1] = (3 * A[:, -1] - 4 * A[:, -2] + A[:, -3]) / (2 * h)
    return D


def apply_boundary_conditions(U):
    """
    Left and bottom walls are solid, so u = 0 there.
    At the outer edge of the PML we also set u = 0.
    If the layer is strong enough, the wave should be tiny there.
    """
    U[0, :] = 0.0
    U[:, 0] = 0.0
    U[-1, :] = 0.0
    U[:, -1] = 0.0


def pml_profile(grid, start, width, sigma_max, power):
    """
    Smooth polynomial profile, zero in the physical region
    and increasing through the PML.
    """
    sigma = np.zeros_like(grid)
    mask = grid > start
    eta = (grid[mask] - start) / width
    sigma[mask] = sigma_max * eta ** power
    return sigma


def pml_rhs(U, V, W, PSI, h, sigma_x, sigma_y):
    """
    Lecture-note PML system, extended to separate x and y damping:

        u_t   = v_x + w_y - (sigma_x + sigma_y) u + psi
        v_t   = u_x - sigma_x v
        w_t   = u_y - sigma_y w
        psi_t = sigma_x w_y + sigma_y v_x - sigma_x sigma_y u

    When sigma_x = sigma_y = 0 this reduces to the standard wave system.
    """
    Ux = dx(U, h)
    Uy = dy(U, h)
    Vx = dx(V, h)
    Wy = dy(W, h)

    sigma_sum = sigma_x[:, None] + sigma_y[None, :]

    U_t = Vx + Wy - sigma_sum * U + PSI
    V_t = Ux - sigma_x[:, None] * V
    W_t = Uy - sigma_y[None, :] * W
    PSI_t = (
        sigma_x[:, None] * Wy
        + sigma_y[None, :] * Vx
        - sigma_x[:, None] * sigma_y[None, :] * U
    )

    return U_t, V_t, W_t, PSI_t


def rk4_step(U, V, W, PSI, h, k, sigma_x, sigma_y):
    """
    One RK4 step for the semi-discrete PML system.
    """
    k1_u, k1_v, k1_w, k1_psi = pml_rhs(U, V, W, PSI, h, sigma_x, sigma_y)

    U2 = U + 0.5 * k * k1_u
    V2 = V + 0.5 * k * k1_v
    W2 = W + 0.5 * k * k1_w
    PSI2 = PSI + 0.5 * k * k1_psi
    apply_boundary_conditions(U2)

    k2_u, k2_v, k2_w, k2_psi = pml_rhs(U2, V2, W2, PSI2, h, sigma_x, sigma_y)

    U3 = U + 0.5 * k * k2_u
    V3 = V + 0.5 * k * k2_v
    W3 = W + 0.5 * k * k2_w
    PSI3 = PSI + 0.5 * k * k2_psi
    apply_boundary_conditions(U3)

    k3_u, k3_v, k3_w, k3_psi = pml_rhs(U3, V3, W3, PSI3, h, sigma_x, sigma_y)

    U4 = U + k * k3_u
    V4 = V + k * k3_v
    W4 = W + k * k3_w
    PSI4 = PSI + k * k3_psi
    apply_boundary_conditions(U4)

    k4_u, k4_v, k4_w, k4_psi = pml_rhs(U4, V4, W4, PSI4, h, sigma_x, sigma_y)

    U_next = U + (k / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
    V_next = V + (k / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    W_next = W + (k / 6) * (k1_w + 2 * k2_w + 2 * k3_w + k4_w)
    PSI_next = PSI + (k / 6) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)

    apply_boundary_conditions(U_next)
    return U_next, V_next, W_next, PSI_next


def solve_2d_pml(N_physical, time_horizon, courant, physical_length=4.0, sigma_max=10.0, sigma_power=3):
    """
    Solve the 2D wave equation with solid walls on x = 0 and y = 0,
    and a PML of width 2 attached to the boundaries x = 4 and y = 4.

    The grid spacing is chosen from the physical box [0,4] x [0,4].
    """
    h = physical_length / N_physical
    total_length = physical_length + pml_width
    N_total = int(round(total_length / h))

    x = np.linspace(0.0, total_length, N_total + 1)
    y = np.linspace(0.0, total_length, N_total + 1)

    k = courant * h
    J = int(np.ceil(time_horizon / k))
    t = np.arange(J + 1) * k

    X, Y = np.meshgrid(x, y, indexing="ij")

    sigma_x = pml_profile(x, physical_length, pml_width, sigma_max, sigma_power)
    sigma_y = pml_profile(y, physical_length, pml_width, sigma_max, sigma_power)

    U = initial_cond(X, Y)
    V = np.zeros_like(U)
    W = np.zeros_like(U)
    PSI = np.zeros_like(U)
    apply_boundary_conditions(U)

    U_list = [U]
    for n in range(J):
        U, V, W, PSI = rk4_step(U, V, W, PSI, h, k, sigma_x, sigma_y)
        U_list.append(U)

    return x, y, t, U_list, sigma_x, sigma_y


def plot_pml_slice(x, y, U_slice, t_value, physical_length=4.0):


    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, U_slice, shading="auto")
    plt.colorbar(label="u(x,y,t)")
    plt.axvline(physical_length, color="w", linestyle="--", linewidth=1.5)
    plt.axhline(physical_length, color="w", linestyle="--", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Wave field with PML at t = {t_value}")
    plt.tight_layout()
    plt.show()


def plot_sigma_profiles(x, y, sigma_x, sigma_y, physical_length=4.0):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x, sigma_x, linewidth=2)
    axes[0].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\sigma_x(x)$")
    axes[0].set_title("PML profile in x")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(y, sigma_y, linewidth=2)
    axes[1].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("y")
    axes[1].set_ylabel(r"$\sigma_y(y)$")
    axes[1].set_title("PML profile in y")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_required_slices(x, y, U_slice, t_value, physical_length=4.0):
    x_quarter_idx = np.argmin(np.abs(x - 0.25))
    x_two_idx = np.argmin(np.abs(x - 2.0))
    y_half_idx = np.argmin(np.abs(y - 0.5))
    y_two_idx = np.argmin(np.abs(y - 2.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(y, U_slice[x_quarter_idx, :], linewidth=2)
    axes[0, 0].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_title(rf"$u(1/4,y,t)$ at $t={t_value}$")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(y, U_slice[x_two_idx, :], linewidth=2)
    axes[0, 1].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_title(rf"$u(2,y,t)$ at $t={t_value}$")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, U_slice[:, y_half_idx], linewidth=2)
    axes[1, 0].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[1, 0].set_title(rf"$u(x,1/2,t)$ at $t={t_value}$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("u")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, U_slice[:, y_two_idx], linewidth=2)
    axes[1, 1].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[1, 1].set_title(rf"$u(x,2,t)$ at $t={t_value}$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison_slices(
    x_pml, y_pml, U_pml, x_part_a, y_part_a, U_part_a, t_value, physical_length=4.0
):
    x_quarter_idx_pml = np.argmin(np.abs(x_pml - 0.25))
    x_two_idx_pml = np.argmin(np.abs(x_pml - 2.0))
    y_half_idx_pml = np.argmin(np.abs(y_pml - 0.5))
    y_two_idx_pml = np.argmin(np.abs(y_pml - 2.0))

    x_quarter_idx_part_a = np.argmin(np.abs(x_part_a - 0.25))
    x_two_idx_part_a = np.argmin(np.abs(x_part_a - 2.0))
    y_half_idx_part_a = np.argmin(np.abs(y_part_a - 0.5))
    y_two_idx_part_a = np.argmin(np.abs(y_part_a - 2.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(y_pml, U_pml[x_quarter_idx_pml, :], linewidth=2, label="PML")
    axes[0, 0].plot(y_part_a, U_part_a[x_quarter_idx_part_a, :], "--", linewidth=2, label="Part A")
    axes[0, 0].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[0, 0].set_title(rf"$u(1/4,y,t)$ at $t={t_value}$")
    axes[0, 0].set_xlabel("y")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(y_pml, U_pml[x_two_idx_pml, :], linewidth=2, label="PML")
    axes[0, 1].plot(y_part_a, U_part_a[x_two_idx_part_a, :], "--", linewidth=2, label="Part A")
    axes[0, 1].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_title(rf"$u(2,y,t)$ at $t={t_value}$")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("u")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(x_pml, U_pml[:, y_half_idx_pml], linewidth=2, label="PML")
    axes[1, 0].plot(x_part_a, U_part_a[:, y_half_idx_part_a], "--", linewidth=2, label="Part A")
    axes[1, 0].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[1, 0].set_title(rf"$u(x,1/2,t)$ at $t={t_value}$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("u")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x_pml, U_pml[:, y_two_idx_pml], linewidth=2, label="PML")
    axes[1, 1].plot(x_part_a, U_part_a[:, y_two_idx_part_a], "--", linewidth=2, label="Part A")
    axes[1, 1].axvline(physical_length, color="k", linestyle="--", linewidth=1)
    axes[1, 1].set_title(rf"$u(x,2,t)$ at $t={t_value}$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


N_physical = 200
time_horizon = 37
courant = 0.6

x, y, t, slices, sigma_x, sigma_y = solve_2d_pml(N_physical=N_physical, time_horizon=time_horizon,
    courant=courant, sigma_max=10.0, sigma_power=3)

x_part_a, y_part_a, t_part_a, U_part_a = solve_2d_leapfrog(
    N=N_physical, time_horizon=time_horizon, courant=courant, Lx=4.0, Ly=4.0
)

plot_sigma_profiles(x, y, sigma_x, sigma_y)

t_end_idx = len(t) - 1
t_quarter_idx = len(t) // 4
t_95_idx = int(0.95 * t_end_idx)

plot_pml_slice(x, y, slices[0], t[0])
plot_pml_slice(x, y, slices[len(t) // 8], t[len(t) // 8])
plot_pml_slice(x, y, slices[len(t) // 4], t[len(t) // 4])
plot_pml_slice(x, y, slices[len(t) // 2], t[len(t) // 2])
plot_pml_slice(x, y, slices[3 * len(t) // 4], t[3 * len(t) // 4])
plot_pml_slice(x, y, slices[7 * len(t) // 8], t[7 * len(t) // 8])
plot_pml_slice(x, y, slices[t_end_idx], t[t_end_idx])

plot_comparison_slices(x, y, slices[t_quarter_idx], x_part_a, y_part_a, 
    U_part_a[(len(t_part_a) - 1) // 4], t[t_quarter_idx])

plot_comparison_slices(x, y, slices[t_95_idx], x_part_a, y_part_a,
    U_part_a[int(0.95 * (len(t_part_a) - 1))], t[t_95_idx],)

plot_required_slices(x, y, slices[t_quarter_idx],  t[t_quarter_idx], physical_length=4.0)
plot_required_slices(x, y, slices[t_95_idx],  t[t_95_idx], physical_length=4.0)