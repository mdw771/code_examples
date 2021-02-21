import ctypes as ct
ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
ct.pythonapi.PyCapsule_GetPointer.argtypes = [ct.py_object, ct.c_char_p]
ptr_type = ct.CFUNCTYPE(ct.c_double, ct.c_double, ct.c_double)
import mod
# Get the function pointer from the capsule "capsule_mul"
handle = ptr_type(ct.pythonapi.PyCapsule_GetPointer(mod.capsule_mul, "mod.capsule_mul"))

# Get the function pointer from the Cython wrapper around the function pointer.
ct.pythonapi.PyCapsule_GetPointer.restype = ct.c_void_p
ct.pythonapi.PyCapsule_GetPointer.argtypes = [ct.py_object, ct.c_char_p]
ct.pythonapi.PyCapsule_GetName.restype = ct.c_char_p
ct.pythonapi.PyCapsule_GetName.argtypes = [ct.py_object]
other_handle = ptr_type(
                   ct.pythonapi.PyCapsule_GetPointer(
                       mod.__pyx_capi__['cymul'],
                       ct.pythonapi.PyCapsule_GetName(
                           mod.__pyx_capi__['cymul'])))

# Call all three exposed versions of the function.
print(handle(ct.c_double(1), ct.c_double(2)))
print(other_handle(ct.c_double(1), ct.c_double(2)))
print(mod.pymul(1, 2))
