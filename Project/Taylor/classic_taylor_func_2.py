# import numpy as np
# import math
#
# from Ex2.Q2_func import _compute_mvdr_weights
#
#
# def compute_mvdr_weights(Rn, d):
#     """
#     Rn: [F, M, M]
#     d : [M, F]
#     """
#
#     F, M, _ = Rn.shape
#     w = np.zeros((F, M), dtype=np.complex128)
#
#     for f in range(F):
#         Rn_inv = np.linalg.pinv(Rn[f])
#         d_f = d[:, f]
#         numerator = Rn_inv @ d_f
#         denominator = d_f.conj().T @ Rn_inv @ d_f
#         w[f] = numerator / denominator
#
#     return w
#
#
# def compute_taylor_beamformer(Rn, d, order=1):
#     """
#     Analytic Taylor beamformer
#
#     Rn: [F, M, M]
#     d : [M, F]
#     """
#
#     F, M, _ = Rn.shape
#     w_mvdr = _compute_mvdr_weights(Rn, d)
#
#     w_taylor = np.zeros_like(w_mvdr)
#
#     for f in range(F):
#
#         Rn_inv = np.linalg.pinv(Rn[f])
#         d_f = d[:, f]
#         w0 = w_mvdr[f]
#
#         denom = d_f.conj().T @ Rn_inv @ d_f
#
#         correction = np.zeros(M, dtype=np.complex128)
#
#         for k in range(1, order+1):
#             coeff = ((-1)**k) / math.factorial(k)
#             term = (Rn_inv @ d_f) / (denom ** (k+1))
#             correction += coeff * term
#
#         w_taylor[f] = w0 + correction
#
#     return w_taylor
