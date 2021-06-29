#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdlib cimport abs
from libc.math cimport sqrt, ceil, M_PI
from cython.parallel import prange

cnp.import_array()


ctypedef fused np_ints:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_uints:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

ctypedef fused np_complexes:
    cnp.complex64_t
    cnp.complex128_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

ctypedef fused np_numeric:
    np_real_numeric
    np_complexes


cdef inline Py_ssize_t round_v(np_floats r) nogil:
    return <Py_ssize_t>(
        (r + <np_floats>0.5) if (r > <np_floats>0.0) else (r - <np_floats>0.5)
    )


def _hough_line_custom(cnp.ndarray img,
                       int numangle,
                       int numrho,
                       int H,
                       int W):

    cdef float[:,:] img_cython = img
    
    cdef float inrho, itheta
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] theta

    irho = int((H*H + W*W)**0.5 + 1) / float((numrho - 1))
    itheta = M_PI / numangle
    theta = np.arange(numangle) * itheta

    # Compute the array of angles and their sine and cosine
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] ctheta
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] stheta

    ctheta = np.cos(theta)/irho
    stheta = np.sin(theta)/irho

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.float32_t] accum
    cdef Py_ssize_t max_distance, offset
    
    accum = np.zeros((numangle, numrho), dtype=np.float32)
    
    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.npy_intp] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef Py_ssize_t nidxs, nthetas, i, j, x, y, accum_idx

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    with nogil:
        for i in prange(nidxs):
            x = x_idxs[i] - W/2
            y = y_idxs[i] - H/2
            
            for j in prange(nthetas):
                accum_idx = round_v((ctheta[j] * x + stheta[j] * y)) + numrho/2
                # accum[j, accum_idx] += img_cython[x, y]
                accum[j, accum_idx] += 1

    return accum

