import numpy as np


class MethodOfCharacteristics:
    """Löst a(x,y,u) u_x + b(x,y,u) u_y = c(x,y,u) mittels Methode der
    Charakteristiken.
    """

    def __init__(self, a, b, c, L, H, s_max, M, N, u_0):
        self.a = a
        self.b = b
        self.c = c
        self.L = float(L)
        self.H = float(H)
        self.s_max = float(s_max)
        self.M = int(M)
        self.N = int(N)
        self.u_0 = u_0

    def solve_moc(self):
        """Erzeugt für j=0..N die Charakteristiken mit Startpunkten (x_j,0)
        und löst die ODEs dx/ds=a, dy/ds=b, du/ds=c mit einem ODE-Solver.

        Rückgabe:
          dict mit 'x0_points' (N+1,), 's_eval' (M+1,), 'X','Y','U' Arrays der
          Form (M+1,N+1) mit NaNs außerhalb gültiger Teile der Trajektorien.
        """
        try:
            from scipy.integrate import solve_ivp
        except Exception:
            raise RuntimeError("scipy wird benötigt.")

        x0_points = np.linspace(0.0, self.L, self.N + 1)
        s_eval = np.linspace(0.0, self.s_max, self.M + 1)

        trajectories = []

        for x0 in x0_points:
            try:
                u0_val = self.u_0(float(x0))
                u0 = float(u0_val)
            except Exception as e:
                raise TypeError("u_0(x) muss einen skalaren Wert zurückgeben und x als Skalar akzeptieren") from e

            def rhs(s, Y):
                x, y, u = Y
                try:
                    ax = float(self.a(float(x), float(y), float(u)))
                    by = float(self.b(float(x), float(y), float(u)))
                    cu = float(self.c(float(x), float(y), float(u)))
                except Exception as e:
                    raise TypeError("a,b,c müssen skalare Ausgaben für skalare x,y,u liefern") from e
                return [ax, by, cu]

            sol = solve_ivp(rhs, (0.0, float(self.s_max)), [float(x0), 0.0, u0], t_eval=s_eval, rtol=1e-6, atol=1e-8)

            Xs = sol.y[0].copy()
            Ys = sol.y[1].copy()
            Us = sol.y[2].copy()

            # Truncate when leaving rectangular domain [0,L]x[0,H]
            valid = (Xs >= 0.0) & (Xs <= self.L) & (Ys >= 0.0) & (Ys <= self.H)
            if not np.all(valid):
                valid_idx = np.where(valid)[0]
                if valid_idx.size == 0:
                    # keine gültigen Punkte entlang dieser Charakteristik -> leere Trajektorie
                    Xs = np.array([], dtype=float)
                    Ys = np.array([], dtype=float)
                    Us = np.array([], dtype=float)
                    s_local = np.array([], dtype=float)
                else:
                    last = valid_idx[-1]
                    Xs = Xs[: last + 1]
                    Ys = Ys[: last + 1]
                    Us = Us[: last + 1]
                    s_local = sol.t[: last + 1]
            else:
                s_local = sol.t

            trajectories.append({'s': s_local, 'x': Xs, 'y': Ys, 'u': Us})

        # Baue regelmäßige Arrays (M+1, N+1) mit NaNs
        K = self.M + 1
        J = self.N + 1
        X = np.full((K, J), np.nan)
        Y = np.full((K, J), np.nan)
        U = np.full((K, J), np.nan)

        for j, traj in enumerate(trajectories):
            lj = traj['x'].shape[0]
            if lj > 0:
                X[:lj, j] = traj['x']
                Y[:lj, j] = traj['y']
                U[:lj, j] = traj['u']

        # Invertierbarkeitsprüfung anhand Flächen
        area_threshold = 1e-8
        max_k = np.full(J, self.M, dtype=int)

        for k in range(0, K - 1):
            for j in range(0, J - 1):
                corners = np.array([X[k, j], Y[k, j], X[k + 1, j], Y[k + 1, j],
                                    X[k + 1, j + 1], Y[k + 1, j + 1], X[k, j + 1], Y[k, j + 1]])
                if np.any(np.isnan(corners)):
                    continue
                p1 = np.array([X[k, j], Y[k, j]])
                p2 = np.array([X[k + 1, j], Y[k + 1, j]])
                p3 = np.array([X[k + 1, j + 1], Y[k + 1, j + 1]])
                p4 = np.array([X[k, j + 1], Y[k, j + 1]])
                v1 = p3 - p1
                v2 = p4 - p2
                det = v1[0] * v2[1] - v1[1] * v2[0]
                signed_area = 0.5 * det
                if abs(signed_area) < area_threshold:
                    max_k[j] = min(max_k[j], k)
                    max_k[j + 1] = min(max_k[j + 1], k)

        # Kürze die Gitterwerte gemäß max_k
        for j in range(J):
            limit = max_k[j]
            if limit < self.M:
                X[limit + 1:, j] = np.nan
                Y[limit + 1:, j] = np.nan
                U[limit + 1:, j] = np.nan

        return {'x0_points': x0_points, 's_eval': s_eval, 'X': X, 'Y': Y, 'U': U}

    def interpolate_solution(self, X, Y, U, nx=201, ny=201, method='linear'):
        """Interpoliert Punkte (X,Y,U) auf ein regelmäßiges Gitter in [0,L]x[0,H].
        Nicht abgedeckte Stellen bleiben NaN. Nur für skalare u gedacht.
        """
        try:
            from scipy.interpolate import griddata
        except Exception:
            raise RuntimeError("scipy.interpolate wird benötigt.")

        mask = ~np.isnan(U)
        if np.count_nonzero(mask) == 0:
            raise ValueError('Keine gültigen Punkte zum Interpolieren vorhanden.')

        pts = np.column_stack((X[mask], Y[mask]))
        vals = U[mask]

        xg = np.linspace(0.0, self.L, nx)
        yg = np.linspace(0.0, self.H, ny)
        Xg, Yg = np.meshgrid(xg, yg, indexing='xy')
        grid_pts = np.column_stack((Xg.ravel(), Yg.ravel()))

        # Versuche robuste Linear-Interpolation, fallback auf nearest
        try:
            from scipy.spatial import Delaunay
            from scipy.interpolate import LinearNDInterpolator
            tri = Delaunay(pts, qhull_options='QJ')
            lin = LinearNDInterpolator(tri, vals, fill_value=np.nan)
            Ug_flat = lin(grid_pts)
            Ug = np.asarray(Ug_flat).reshape((ny, nx))
            # Punkte außerhalb auf NaN setzen
            simplex = tri.find_simplex(grid_pts)
            in_hull = simplex >= 0
            Ug_flat = Ug.ravel()
            Ug_flat[~in_hull] = np.nan
            Ug = Ug_flat.reshape((ny, nx))
        except Exception:
            Ug = griddata(pts, vals, grid_pts, method=method, fill_value=np.nan)
            Ug = Ug.reshape((ny, nx))

        return xg, yg, Ug


if __name__ == '__main__':
    # Setze gemeinsame Parameter
    import math

    # Standardwerte
    N = 100
    M = 100
    nx = 201
    ny = 201

    def run_case(label, a, b, c, u0, L, H, s_max, exact_func, note='', y_max_valid=None):
        moc = MethodOfCharacteristics(a, b, c, L, H, s_max, M, N, u0)
        res = moc.solve_moc()
        X = res['X']
        Y = res['Y']
        U = res['U']
        print(f'Case {label}: gültige MoC-Punkte =', int(np.count_nonzero(~np.isnan(U))))
        try:
            xg, yg, Ug = moc.interpolate_solution(X, Y, U, nx=nx, ny=ny, method='linear')
        except Exception as e:
            print(f'Case {label}: Interpolation fehlgeschlagen:', e)
            return

        Xg, Yg = np.meshgrid(xg, yg, indexing='xy')
        U_exact = exact_func(Xg, Yg)

        # Maskiere Bereiche, die nicht gültig sind (z.B. y >= 1 in Aufgabe b)
        if y_max_valid is not None:
            mask_invalid = Yg >= float(y_max_valid)
            Ug[mask_invalid] = np.nan
            U_exact[mask_invalid] = np.nan

        mask = ~np.isnan(Ug) & ~np.isnan(U_exact)
        if np.any(mask):
            diff = Ug[mask] - U_exact[mask]
            l2 = np.sqrt(np.mean(diff ** 2))
            linf = np.max(np.abs(diff))
            print(f'Case {label}: Fehler (L2) = {l2:.6e}, (Linf) = {linf:.6e} auf {np.count_nonzero(mask)} Punkten')
        else:
            print(f'Case {label}: Keine interpolierten Punkte zum Vergleich vorhanden.')

        # Plot figure für den Fall
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print('Matplotlib wird zum Plotten benötigt.')
            return

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'{label}) {note}', fontsize=12)

        im0 = axs[0].imshow(Ug, origin='lower', extent=(0, L, 0, H), aspect='auto')
        axs[0].set_title(f'{label}) MoC Interpolation')
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(U_exact, origin='lower', extent=(0, L, 0, H), aspect='auto')
        axs[1].set_title(f'{label}) Exakte Lösung')
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(Ug - U_exact, origin='lower', extent=(0, L, 0, H), aspect='auto', cmap='bwr')
        axs[2].set_title(f'{label}) Differenz (MoC - Exact)')
        fig.colorbar(im2, ax=axs[2])

        for ax in axs:
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Aufgabe a)
    L_a = math.pi
    H_a = math.pi
    s_max_a = H_a
    def a_a(x, y, u): return 1.0
    def b_a(x, y, u): return 2.0
    def c_a(x, y, u): return 0.0
    def u0_a(x): return math.sin(float(x))
    def exact_a(X, Y): return np.sin(X - 0.5 * Y)
    note_a = "u_x + 2 u_y = 0, u(x,0)=sin(x), exact: u(x,y)=sin(x - y/2)"

    # Aufgabe b)
    L_b = 2.0
    H_b = 2.0
    s_max_b = H_b
    def a_b(x, y, u):
        return float(u)
    def b_b(x, y, u):
        return 1.0
    def c_b(x, y, u):
        return 0.0
    def u0_b(x): return -float(x)
    def exact_b(X, Y):
        with np.errstate(divide='ignore', invalid='ignore'):
            return -X / (1.0 - Y)
    note_b = "u * u_x + u_y = 0, u(x,0)=-x, exact: u=-x/(1-y) (nur für 0<=y<1)"

    # Run both cases and show two figures with Überschriften a) and b)
    run_case('a', a_a, b_a, c_a, u0_a, L_a, H_a, s_max_a, exact_a, note_a)
    run_case('b', a_b, b_b, c_b, u0_b, L_b, H_b, s_max_b, exact_b, note_b, y_max_valid=1.0)

