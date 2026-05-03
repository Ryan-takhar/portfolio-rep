import numpy as np
import matplotlib.pyplot as plt


delta = 1/5
xp = 1/2
yp = 1


def initial_cond(X, Y):
    """
    The initial conditions are
    u(x,y,0) = cos(pi r / (2 delta))   for r <= delta
                 = 0                       otherwise
    where r = sqrt((x-xp)^2 + (y-yp)^2
    """
    r = np.sqrt((X-xp)**2 + (Y-yp)**2)

    U_0 = np.zeros_like(X)
    labels = r <= delta
    U_0[labels] = np.cos(np.pi * r[labels] / (2 * delta))
    return U_0


def solve_2d_leapfrog(N, time_horizon, courant, Lx, Ly):
    """
    We are trying to solve u_tt = u_xx + u_yy 
    on [0, Lx] x [0, Ly] subject to many boundary
    conditions
    
    N: Number of intervals in each spatial direction.
    So there are N + 1 grid points in each direction.
    
    time_horizon: Final time to simulate to.

    courant: equal to k / h

    Lx, Ly: Domain Lengths in x and y
    xp, yp: Centre of the initial pulse

    Returns 
    x : (N+1,) ndarray
    y : (N+1,) ndarray
    t : (J+1,) ndarray
    U : (J+1, N+1, N+1) ndarray
        U[n, i, j] approximates u(x_i, y_j, t^n).
    """


    x = np.linspace(0, Lx, N+1)
    y = np.linspace(0, Ly, N+1)

    h = x[1] - x[0]
    k = courant * h

    J = int(np.ceil(time_horizon / k))
    t = np.arange(J + 1) * k

    U = np.zeros((J+1, N+1, N+1))
    X, Y = np.meshgrid(x, y, indexing='ij')


    U[0] = initial_cond(X, Y)

    # Enforce solid walls at t = 0
    U[0, 0, :] = 0.0      # x = 0
    U[0, :, 0] = 0.0      # y = 0

    # --------------------------------------------------
    # Start-up step: compute U[1] using Taylor expansion
    # since u_t(x,y,0) = 0
    # --------------------------------------------------
    U[1, 1:N, 1:N] = (
        U[0, 1:N, 1:N]
        + 0.5 * courant**2 * (U[0, 2:N+1, 1:N] - 2 * U[0, 1:N, 1:N] + U[0, 0:N-1, 1:N])
        + 0.5 * courant**2 * (U[0, 1:N, 2:N+1] - 2 * U[0, 1:N, 1:N] + U[0, 1:N, 0:N-1])
    )

    # Solid boundaries at first step
    U[1, 0, :] = 0.0
    U[1, :, 0] = 0.0

    # Right boundary startup: u_t + u_x = 0
    # (forward in time, backward in x)
    U[1, N, 1:N] = (
        U[0, N, 1:N]
        + 0.25 * courant**2 * (
            U[0, N, 2:N+1] - 2 * U[0, N, 1:N] + U[0, N, 0:N-1]
        )
    )

    # Top boundary startup: u_t + u_y = 0
    # (forward in time, backward in y)
    U[1, 1:N, N] = (
        U[0, 1:N, N]
        + 0.25 * courant**2 * (
            U[0, 2:N+1, N] - 2 * U[0, 1:N, N] + U[0, 0:N-1, N]
        )
    )

    for n in range(1, J):
        # Interior leapfrog update
        U[n + 1, 1:N, 1:N] = (
            2 * U[n, 1:N, 1:N]
            - U[n - 1, 1:N, 1:N]
            + courant **2 * (
                U[n, 2:N+1, 1:N] - 2 * U[n, 1:N, 1:N] + U[n, 0:N-1, 1:N]
                + U[n, 1:N, 2:N+1] - 2 * U[n, 1:N, 1:N] + U[n, 1:N, 0:N-1]
            )
        )

        # Solid boundaries
        U[n + 1, 0, :] = 0.0
        U[n + 1, :, 0] = 0.0

        # ---------------------------------
        # Top boundary y = 4
        #
        # u_yt = -u_tt + (1/2)u_xx
        #
        # This gives:
        #
        # U^{n+1}_{i,N}
        # =
        # [ 2U^n_{i,N}
        #   - (1 - q/4)U^{n-1}_{i,N}
        #   + (q^2/2) delta_xx U^n_{i,N}
        #   + (q/4)(U^{n+1}_{i,N-1} - U^{n-1}_{i,N-1})
        # ] / (1 + q/4)
        # ---------------------------------
        lap_x_top = U[n, 2:N+1, N] - 2 * U[n, 1:N, N] + U[n, 0:N-1, N]

        U[n + 1, 1:N, N] = (
            2 * U[n, 1:N, N]
            - (1 - courant / 4) * U[n - 1, 1:N, N]
            + 0.5 * courant **2 * lap_x_top
            + (courant / 4) * (U[n + 1, 1:N, N - 1] - U[n - 1, 1:N, N - 1])
        ) / (1 + courant / 4)

        # ---------------------------------
        # Right boundary x = 4
        #
        # u_xt = -u_tt + (1/2)u_yy
        # ---------------------------------
        lap_y_right = U[n, N, 2:N+1] - 2 * U[n, N, 1:N] + U[n, N, 0:N-1]

        U[n + 1, N, 1:N] = (
            2 * U[n, N, 1:N]
            - (1 - courant / 4) * U[n - 1, N, 1:N]
            + 0.5 * courant**2 * lap_y_right
            + (courant / 4) * (U[n + 1, N - 1, 1:N] - U[n - 1, N - 1, 1:N])
        ) / (1 + courant / 4)

        # Remaining corners
        U[n + 1, N, 0] = 0.0
        U[n + 1, 0, N] = 0.0

        # Simple practical closure at top-right corner
        U[n + 1, N, N] = 0.5 * (U[n + 1, N - 1, N] + U[n + 1, N, N - 1])

    return x, y, t, U

def leapfrog_plots(x, y, t_end, U_snapshot):
    """
    plots how the solutions
    evolve with time along the 
    the axes (x, y = 0) and (x = 0, y)
    """
    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(6, 5))

    plt.pcolormesh(X, Y, U_snapshot, shading="auto")
    plt.colorbar(label="u(x,y,t)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    print(t_end)
    plt.show()



if __name__ == "__main__":
    N = 200
    time_horizon = 37
    courant = 0.6
    Lx, Ly = 4, 4
    x, y, t, U = solve_2d_leapfrog(N, time_horizon, courant, Lx, Ly)

    leapfrog_plots(x, y, t_end = t[0], U_snapshot=U[0])

    leapfrog_plots(x, y, t_end=t[len(t)//8], U_snapshot=U[len(t)//8])

    leapfrog_plots(x, y, t_end=t[len(t)//4], U_snapshot=U[len(t)//4])
    # Plot a middle snapshot
    leapfrog_plots(x, y, t_end=t[len(t)//2], U_snapshot=U[len(t)//2])

    leapfrog_plots(x, y, t_end=t[3*len(t)//4], U_snapshot=U[3*len(t)//4])

    leapfrog_plots(x, y, t_end=t[7*len(t)//8], U_snapshot=U[7* len(t)//8])
    # # Plot final snapshot

    leapfrog_plots(x, y, t_end=t[-1], U_snapshot=U[-1])

    # ============================================================
    # Question 2: self-convergence using the existing solver
    # ===========================================================

    def restrict_to_coarse(U_fine):
        """
        Restrict a fine-grid snapshot to the coarse grid
        by taking every other grid point.

        This assumes the fine grid has twice as many intervals
        as the coarse grid.
        """
        return U_fine[::2, ::2]


    def l2_error(U_coarse, U_fine_restricted, h):
        """
        Discrete L2 error on the coarse grid.
        """
        E = U_coarse - U_fine_restricted
        return np.sqrt(h**2 * np.sum(E**2)), E


    def contour_plot(x, y, Z, title, levels=20):
        X, Y = np.meshgrid(x, y, indexing='ij')

        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, Z, levels=levels)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def refinement_study(N_values, courant):
        """
        Run the solver for several N values and compare successive pairs:
            N, 2N, 4N, ...

        The time_horizon is chosen so that all grids land exactly
        on the same physical time.

        Parameters
        ----------
        N_values : list of ints
            Should ideally be something like [50, 100, 200, 400]
        n_steps_base : int
            Number of time steps on the coarsest grid
        courant : float
            k/h
        """

        N0 = N_values[0]
        h0 = Lx / N0
        k0 = courant * h0
        time_horizon = 2.88

        solutions = {}

        # Compute U_h for each N
        for N in N_values:
            x, y, t, U = solve_2d_leapfrog(N, time_horizon, courant, Lx, Ly)
            solutions[N] = (x, y, t, U[-1])   # store final snapshot only

        # Compare successive pairs
        for i in range(len(N_values) - 1):
            N_coarse = N_values[i]
            N_fine = N_values[i + 1]

            x_c, y_c, t_c, Uc = solutions[N_coarse]
            x_f, y_f, t_f, Uf = solutions[N_fine]

            Uf_restricted = restrict_to_coarse(Uf)

            h = x_c[1] - x_c[0]
            err_l2, E = l2_error(Uc, Uf_restricted, h)

            print(f"N = {N_coarse} vs {N_fine} | time = {t_c[-1]} | L2 error = {err_l2}")

            contour_plot(
                x_c, y_c, E,
                title=f"Error contour: N={N_coarse} vs N={N_fine} at t={t_c[-1]}",
                levels=20
            )

        return solutions

    N_values = [50, 100, 200, 400]
    solutions = refinement_study(N_values=N_values, courant=courant)

    # ============================================================
    #                          Question 4
    # ===========================================================

    T_end = 37

    t_end_idx = len(t) - 1
    t_quarter_idx = t_end_idx // 4
    t_95_idx = int(0.95 * t_end_idx)

    x_quarter_idx = np.argmin(np.abs(x - 0.25))
    x_two_idx = np.argmin(np.abs(x - 2.0))
    y_half_idx = np.argmin(np.abs(y - 0.5))
    y_two_idx = np.argmin(np.abs(y - 2.0))


    def plot_required_slices(U_snapshot, t_value):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(y, U_snapshot[x_quarter_idx, :], linewidth=2)
        axes[0, 0].set_title(rf"$u(1/4,y,t)$ at $t={t_value}$")
        axes[0, 0].set_xlabel("y")
        axes[0, 0].set_ylabel("u")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(y, U_snapshot[x_two_idx, :], linewidth=2)
        axes[0, 1].set_title(rf"$u(2,y,t)$ at $t={t_value}$")
        axes[0, 1].set_xlabel("y")
        axes[0, 1].set_ylabel("u")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(x, U_snapshot[:, y_half_idx], linewidth=2)
        axes[1, 0].set_title(rf"$u(x,1/2,t)$ at $t={t_value}$")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("u")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(x, U_snapshot[:, y_two_idx], linewidth=2)
        axes[1, 1].set_title(rf"$u(x,2,t)$ at $t={t_value}$")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("u")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    plot_required_slices(U[t_quarter_idx], t[t_quarter_idx])
    plot_required_slices(U[t_95_idx], t[t_95_idx])


