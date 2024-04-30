import numpy as np
import scipy.io as sio


def load_grid_data(filename):
    data = sio.loadmat(filename)
    Y = data['Y']
    N = Y.shape[0]
    return Y, N


def generate_data_grid(P, grid, beta=3, c_ww=0):
    """
    @P: no. of samples
    @Lambda: eigenvalue matrix
    @V: eigenvectors matrix
    @beta: default 3
    @c_ww: var. of noise
    return: x_p as row vectors in a matrix
    """
    N = grid.Y.shape[0]
    y_p = np.zeros((P, N))

    x_p = grid.generate_x_p(P, beta)
    for p in range(0, P):
        y_p[p] = grid.g_x_L(x_p[p])
    y_p += np.sqrt(c_ww) * np.random.randn(P, N)

    return x_p, y_p


def calc_dxy_dyy(x_p, g_p, V, c_ww):
    x_p = x_p.T     # column vectors representation
    g_p = g_p.T     # column vectors representation

    x_p_mean = np.mean(x_p, axis=1)[:, np.newaxis]
    y_p_mean = np.mean(g_p, axis=1)[:, np.newaxis]

    d_xy = np.mean( ( V.T @ (x_p - x_p_mean) ) * ( V.T @ (g_p - y_p_mean) ) , axis=1)

    d_yy = np.mean( ( V.T @ (g_p - y_p_mean) ) ** 2 , axis=1) + c_ww

    return d_xy, d_yy


def calc_gsp_lmmse_estim(x_p, g_p, grid, c_ww):
    """
    @x_p, g_p: data
    @V: array of eigenvectors matrix
    @c_ww: var. of noise
    return: H of (Sample)_GSP_LMMSE
    """    
    d_xy, d_yy = calc_dxy_dyy(x_p, g_p, grid.V, c_ww)

    H_GSP = np.diag(d_xy / d_yy)

    return H_GSP


def calc_sLMMSE_estimator(x_p, g_p, c_ww):
    P = x_p.shape[0]
    N = x_p.shape[1]
    x_p = x_p.T  # column vectors representation
    g_p = g_p.T  # column vectors representation

    x_p_mean = np.mean(x_p, axis=1)[:, np.newaxis]
    y_p_mean = np.mean(g_p, axis=1)[:, np.newaxis]

    C_xy = (x_p - x_p_mean) @ (g_p - y_p_mean).T / P
    C_yy = ((g_p - y_p_mean) @ (g_p - y_p_mean).T + c_ww * np.eye(N)) / P

    A_sLMMSE = C_xy @ np.linalg.inv(C_yy)
    return A_sLMMSE


def calc_aLMMSE_estimator(L, c_ww, beta=3):
    N = L.shape[0]

    A_aLMMSE = beta * np.linalg.pinv(L) @ L @ np.linalg.inv(beta * L + c_ww * np.eye(N))

    return A_aLMMSE


def test_estimator_mse(H, grid, x_p_test, y_p_test):
    """
    @H: matrix of freq. coeff. of the estimator
    @V: graph fourier basis, i.e. eigenvectors
    @x_p_test, y_p_test: data for testing
    return MSE
    """
    x_p_test = x_p_test.T     # column vectors representation
    y_p_test = y_p_test.T     # column vectors representation

    V = grid.V

    err = V @ H @ V.T @ y_p_test - x_p_test
    s_mse = np.linalg.norm(err, axis=0)
    mse = np.mean(s_mse)
    return mse


def test_estimator_rand_laplacian_mse_H(H, grid_m, x_p_m_test, y_p_m_test, p_m):
    """
    @H: matrix of freq. coeff. of the estimator
    @V_m: graph fourier basis, of each laplacian
    @x_p_m_test, y_p_m_test: data
    @p_m: the prob. of each laplacian
    return MSE
    """
    M = len(grid_m)
    mse_m = np.zeros(M)

    for m in range(0, M):
        V = grid_m[m].V
        mse_m[m] = test_estimator_mse_L_init_A(V @ H @ V.T, x_p_m_test[m], y_p_m_test[m])

    mse = mse_m.dot(p_m)

    return mse


def test_estimator_mse_L_init_A(A, x_p_test, y_p_test):
    """
    @H: matrix of freq. coeff. of the estimator
    @V: graph fourier basis, i.e. eigenvectors
    @x_p_test, y_p_test: data for testing
    return MSE
    """
    x_p_test = x_p_test.T     # column vectors representation
    y_p_test = y_p_test.T     # column vectors representation

    err = A @ y_p_test - x_p_test
    s_mse = np.linalg.norm(err, axis=0)
    mse = np.mean(s_mse)
    return mse


def vander_matrix(v, Q):
    Psi = np.array([v] * Q).T ** range(Q)

    return Psi
