import numpy as np


class Grid:
    """
    This class represents the attributes of a power grid
    """
    def __init__(self, Y):
        """
        :param Y: Grid's admittance matrix Y = G + iB
        """
        self.Y = Y
        self.L = - np.imag(Y)
        self.Lambda, self.V = evd(self.L)
        self.__cal_mean_std_G_B()

    def g_x_L(self, x):
        """
        This funtion returns g(x;L) s.t. [g(x;L)]_i = sum_j [L]_i,j * sin(x_i - x_j)
        :param x: an input signal
        :return: g(x;L)
        """
        v = np.exp(1j * x)
        g_x = np.real(v * np.conj(self.Y @ v))

        return g_x

    def generate_x_p(self, P, beta=3):
        """
        @P: no. of samples
        @beta: default 3
        return: x_p as row vectors in a matrix
        """
        N = self.Y.shape[0]
        x_p = (self.V[:, 1:] @ np.random.multivariate_normal(np.zeros(N - 1), beta * np.diag(1 / np.diag(self.Lambda)[1:]), P).T).T
        if P == 1:
            x_p = x_p[0, :]
        return x_p

    def generate_random_grid(self, prob_disconn, prob_conn):
        N = self.Y.shape[0]

        # Find unconnected edges
        temp = self.Y + np.tri(N)
        unconnected = np.where(temp == 0)

        # Find connected edges
        temp = np.copy(self.Y)
        temp[np.tri(N) == 1] = 0
        connected = np.where(temp != 0)

        t = 0
        while True:
            Y_mod = np.copy(self.Y)

            for idx in range(len(unconnected[0])):
                if np.random.rand() <= prob_conn:
                    i, j = unconnected[0][idx], unconnected[1][idx]
                    g_ij = -abs(np.random.normal(self.mean_G, self.std_G))
                    b_ij = abs(np.random.normal(self.mean_B, self.std_B))
                    y_ij = g_ij + 1j * b_ij
                    Y_mod[i, i] -= y_ij
                    Y_mod[j, j] -= y_ij
                    Y_mod[i, j] += y_ij
                    Y_mod[j, i] += y_ij

            for idx in range(len(connected[0])):
                if np.random.rand() <= prob_disconn:
                    i, j = connected[0][idx], connected[1][idx]
                    y_ij = Y_mod[i, j]
                    Y_mod[i, i] += y_ij
                    Y_mod[j, j] += y_ij
                    Y_mod[i, j] = 0
                    Y_mod[j, i] = 0

            grid_mod = Grid(Y_mod)
            if grid_mod.is_connected():
                return grid_mod
            elif t > 50:  # time-out
                raise ValueError("Generating connected graph failed")
            t += 1

    def is_connected(self):
        return abs(self.Lambda[1, 1]) > 10 ** -5

    def __cal_mean_std_G_B(self):
        N = self.Y.shape[0]

        arr = self.Y[np.tri(N) == 0]
        arr = arr[arr != 0]

        self.mean_G = np.mean(np.real(arr))
        self.std_G = min(abs(self.mean_G)/3., np.std(np.real(arr)))

        self.mean_B = np.mean(np.imag(arr))
        self.std_B = min(abs(self.mean_B)/3., np.std(np.imag(arr)))
        # std s.t. the mean is at least 3 std away from zero, i.e. ~99% chance a positive value will be sampled


def evd(A):
    """
    Eigen value decomposition of the Laplacian matrix
    :param A: an input matrix
    :return: Lambda = diag(lambda), V - graph fourier basis
    """
    Lambda, V = np.linalg.eig(A)
    sorted_indices = np.argsort(Lambda)
    Lambda = Lambda[sorted_indices]
    Lambda = np.diag(Lambda)
    V = V[:, sorted_indices]
    return Lambda, V
