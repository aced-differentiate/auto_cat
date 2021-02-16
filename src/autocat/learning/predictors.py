import qml.kernels as qml_kernels
import qml.math as qml_math
import numpy as np


def get_alphas(
    kernel: np.ndarray,
    perturbation_matrices: np.ndarray,
    regularization_strength: float = 1e-8,
):
    """
    Wrapper for `qml.math.cho_solve` to
    solve for the regression coefficients using cholesky
    decomposition (ie. training a KRR)

    alphas = (K + lambda*I)^(-1)y

    Parameters
    ----------

    kernel:
        Numpy array of kernel matrix (e.g. generated via `get_kernel`)

    perturbation_matrices:
        Numpy array containing perturbation matrices corresponding to the
        input structures which form the labels when training

    regularization_strength:
        Float specifying regularization strength (lambda in eqtn above)

    perturbation

    Returns
    -------

    alphas:
        Numpy array of regression coefficients

    """
    K = kernel + regularization_strength * np.identity(kernel.shape)

    return qml_math.cho_solve(K, perturbation_matrices)


def get_kernel(X1: np.ndarray, X2: np.ndarray, kernel_type: str = "gaussian", **kwargs):
    """
    Wrapper for `qml` kernels

    Parameters
    ----------

    X1,X2:
         Numpy arrays to be used to calculate the kernel

    kernel_type:
        String giving type of kernel to be generated.
        Options:
            - gaussian (default)
            - laplacian
            - linear
            - matern
            - sargan

    Returns
    -------

    kernel_matrix:
        Numpy array of generated kernel matrix
    """
    kernel_funcs = {
        "laplacian": qml_kernels.laplacian_kernel,
        "gaussian": qml_kernels.gaussian_kernel,
        "linear": qml_kernels.linear_kernel,
        "matern": qml_kernels.matern_kernel,
        "sargan": qml_kernels.sargan_kernel,
    }
    return kernel_funcs[kernel_type](X1, X2, **kwargs)
