cimport cython
from cython.parallel cimport prange

@cython.cdivision(True)
cdef float lerp(float a, float b, float t) nogil:
    return (b - a) * t + a

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef cython.uchar lerp2d(cython.uchar [:, :] input,
                      float i, float j) nogil:
    cdef int inti, intj
    cdef float deltai, deltaj

    inti = <int> i
    intj = <int> j
    if inti == input.shape[0] - 1:
        inti -= 1
    if intj == input.shape[1] - 1:
        intj -= 1
    deltai = i - inti
    deltaj = j - intj

    return <cython.uchar> lerp(lerp(<double> input[inti, intj], <double>input[inti + 1, intj], deltai),
                            lerp(<double>input[inti, intj + 1], <double> input[inti + 1, intj + 1], deltai),
                            deltaj)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _remap(cython.uchar [:, :] input,
                 cython.uchar [:, :] output,
                 double [:, :] R, double [:, :] T,
                 bint repeat,
                 int base_i, int base_j,
                 int offset_i, int offset_j) nogil:
   cdef int i, j, ti, tj
   cdef float interped_row, interped_col

   for i in prange(output.shape[0], schedule='static', num_threads=16):
       for j in range(output.shape[1]):
           # apply rigid transformation
           ti = i - base_i - offset_i
           tj = j - base_j - offset_j
           interped_row = ti * R[0, 0] + tj * R[0, 1] + T[0, 0] + offset_i
           interped_col = ti * R[1, 0] + tj * R[1, 1] + T[1, 0] + offset_j

           # deal with clamping and repeat
           if interped_row < 0:
               if not repeat:
                   continue
               interped_row = 0
           elif interped_row > input.shape[0] - 1:
               if not repeat:
                   continue
               interped_row = input.shape[0] - 1
           if interped_col < 0:
               if not repeat:
                   continue
               interped_col = 0
           elif interped_col > input.shape[1] - 1:
               if not repeat:
                   continue
               interped_col = input.shape[1] - 1

           output[i, j] = lerp2d(input, interped_row, interped_col)

cpdef remap(cython.uchar [:, :] input,
            cython.uchar [:, :] output,
            double [:, :] R, double [:, :] T,
            bint repeat,
            base_i=0, base_j=0,
            offset_i=0, offset_j=0):
    _remap(input, output, R, T, repeat, base_i, base_j, offset_i, offset_j)

