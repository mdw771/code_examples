# distutils: language = c++

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

cdef extern from "ops.hpp":
    cdef double mul(double a, double b) nogil

capsule_mul = PyCapsule_New(<void*>&mul, "mod.capsule_mul", NULL)

cdef double (*mul_ptr)(double, double) nogil
mul_ptr = <double (*)(double, double) nogil> PyCapsule_GetPointer(capsule_mul, "mod.capsule_mul")

cdef double cymul(double a, double b) nogil:
    return mul_ptr(a, b)

def pymul(double a, double b):
    return cymul(a, b)
