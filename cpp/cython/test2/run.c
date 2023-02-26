#include <Python.h>
#include "py_c.h"

int main()
{
    Py_Initialize();
    PyInit_py_c();
    boom();
    return 0;
}
