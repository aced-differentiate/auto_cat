import qml.kernels as qml_kernels
import numpy as np


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
