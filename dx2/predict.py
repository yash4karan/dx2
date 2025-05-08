import math
import matplotlib.pyplot as plt
import numpy as np


def calc_h_lims(T):
    a0 = T[0, 2] ** 2 - T[0, 0] * T[2, 2]
    b0 = 2 * (T[0, 2] * T[2, 3] - T[0, 3] * T[2, 2])
    c0 = T[2, 3] ** 2
    b1 = T[0, 2] * T[1, 2] - T[0, 1] * T[2, 2]
    c1 = T[1, 2] * T[2, 3] - T[1, 3] * T[2, 2]
    c2 = T[1, 2] ** 2 - T[1, 1] * T[2, 2]

    if ((2 * b1 * c1 - b0 * c2) / (2 * (a0 * c2 - b1**2))) ** 2 + (c0 * c2 - c1**2) / (
        a0 * c2 - b1**2
    ) >= 0:
        a = (2 * b1 * c1 - b0 * c2) / (2 * (a0 * c2 - b1**2)) - np.sqrt(
            ((2 * b1 * c1 - b0 * c2) / (2 * (a0 * c2 - b1**2))) ** 2
            + (c0 * c2 - c1**2) / (a0 * c2 - b1**2)
        )
        b = (2 * b1 * c1 - b0 * c2) / (2 * (a0 * c2 - b1**2)) + np.sqrt(
            ((2 * b1 * c1 - b0 * c2) / (2 * (a0 * c2 - b1**2))) ** 2
            + (c0 * c2 - c1**2) / (a0 * c2 - b1**2)
        )
        return (math.ceil(a), math.floor(b))


def calc_k_lims(T, h):
    r0 = T[2, 3] ** 2 + h * (
        2 * (T[0, 2] * T[2, 3] - T[0, 3] * T[2, 2])
        + h * (T[0, 2] ** 2 - T[0, 0] * T[2, 2])
    )
    r1 = (
        T[1, 2] * T[2, 3]
        - T[1, 3] * T[2, 2]
        + h * (T[0, 2] * T[1, 2] - T[0, 1] * T[2, 2])
    )
    r2 = T[1, 2] ** 2 - T[1, 1] * T[2, 2]

    a = (-r1 + np.sqrt(r1**2 - r0 * r2)) / r2
    b = (-r1 - np.sqrt(r1**2 - r0 * r2)) / r2
    return tuple(sorted([round(a), round(b)]))


def calc_l(T, h, k):
    q0 = (
        T[0, 0] * h**2
        + 2 * T[0, 1] * h * k
        + T[1, 1] * k**2
        + 2 * T[0, 3] * h
        + 2 * T[1, 3] * k
    )
    q1 = T[0, 2] * h + T[1, 2] * k + T[2, 3]
    q2 = T[2, 2]
    if q1**2 - q0 * q2 >= 0:
        a = (-q1 - np.sqrt(q1**2 - q0 * q2)) / q2
        b = (-q1 + np.sqrt(q1**2 - q0 * q2)) / q2
        lst = []
        if abs(round(a) - a) < 0.1:
            lst.append(round(a))
        if abs(round(b) - b) < 0.1:
            lst.append(round(b))
        return tuple(lst)


if __name__ == "__main__":
    # edge length of cubic lattice
    a = 10
    # detector sensor matrix (pixel counts)
    sensor_matrix = np.ndarray((1000, 1000))
    # incoming wave vector
    s = 5
    s_vec = np.array([-s, 0, 0])
    # spindle axis
    e_vec = np.array([0, 1, 0])
    # crystal setting matrix
    A = 1 / a * np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # angular resolution (radians)
    d_phi = 1 * np.pi / 180
    # d-matrix
    d = np.matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # D-matrix
    D = np.linalg.inv(d)

    # alpha = np.matmul(np.transpose(s_vec), A)
    # beta = np.matmul(np.transpose(np.cross(s_vec, e_vec)), A)
    # gamma = np.dot(e_vec, s_vec) * np.matmul(np.transpose(e_vec), A)
    # delta = 1 / 2 * np.matmul(np.transpose(A), A)
    # # print(f"sensor_matrix:\n{sensor_matrix}")
    # print(f"s_vec:\n{s_vec}")
    # print(f"e_vec:\n{e_vec}")
    # print(f"A:\n{A}")
    # print(f"d_phi: {d_phi}")
    # print(f"alpha: {alpha}")
    # print(f"beta: {beta}")
    # print(f"gamma: {gamma}")
    # print(f"delta:\n{delta}")

    # # hkl vector
    # h = np.array([1, 2, 3])
    # alpha_h = np.matmul(alpha, h).item(0, 0)
    # beta_h = np.matmul(beta, h).item(0, 0)
    # gamma_h = np.matmul(gamma, h).item(0, 0)
    # h_delta_h = np.matmul(np.matmul(np.transpose(h), delta), h).item(0, 0)
    # s = np.sqrt((alpha_h - gamma_h) ** 2 + beta_h**2)

    # a = (h_delta_h - gamma_h) / s
    # b = beta_h / s
    # phi = (np.arcsin(a) - np.arccos(b)) * 180 / np.pi

    # print(f"alpha h  : {alpha_h}")
    # print(f"beta h   : {beta_h}")
    # print(f"gamma h  : {gamma_h}")
    # print(f"h delta h: {h_delta_h}")
    # print(f"  s: {s}")
    # print(f"phi: {phi}")

    R = np.matrix(
        [
            [np.cos(d_phi), 0, np.sin(d_phi)],
            [0, 1, 0],
            [-np.sin(d_phi), 0, np.cos(d_phi)],
        ]
    )

    pixel_phi_list = set()

    for i in range(360):
        pixel_set = set()
        P = np.matrix(
            [
                [-A[0, 0], -A[0, 1], -A[0, 2], s],
                [A[1, 0], A[1, 1], A[1, 2], 0],
                [A[2, 0], A[2, 1], A[2, 2], 0],
            ]
        )

        T = P.T @ P

        h = 1
        k = 1
        l = calc_l(T, h, k)

        index_list = set()
        h_lims = calc_h_lims(T)
        if h_lims is not None:
            for h in range(h_lims[0], h_lims[1] + 1):
                k_lims = calc_k_lims(T, h)
                for k in range(k_lims[0], k_lims[1] + 1):
                    ls = calc_l(T, h, k)
                    if ls is not None:
                        for l in ls:
                            index_list.add((h, k, l))
            for h, k, l in index_list:
                t = np.matrix([s_vec[0], s_vec[1], s_vec[2]]) + A @ np.array([h, k, l])
                X = D @ t.T
                if t[0, 0] >= 0 and X[2, 0] != 0:
                    if (
                        -100000 <= float(X[0, 0] / X[2, 0]) <= 100000
                        and -100000 <= float(X[1, 0] / X[2, 0]) <= 100000
                    ):
                        pixel_set.add(
                            (
                                float(X[0, 0] / X[2, 0]),
                                float(X[1, 0] / X[2, 0]),
                            )
                        )
                        pixel_phi_list.add(
                            (
                                float(X[0, 0] / X[2, 0]),
                                float(X[1, 0] / X[2, 0]),
                                i * d_phi,
                            )
                        )
        data = np.array(list(pixel_set))
        dT = data.T
        if dT.size != 0:
            x, y = dT
            fig, ax = plt.subplots(1, 1)
            ax.plot(x, y, "ro")
            fig.show()
        if dT.size != 0:
            fig, ax = plt.subplots(1, 1)
            fig.show()

        A = R @ A

    pixel_phi_list = sorted(list(pixel_phi_list), key=lambda x: x[2])

    phi = 0
