Converting a Python script containing TensorFlow to a C++ file that contains a main function, so that
it runs on its own. 

Make sure the correct Python library path is prepended to `LD_LIBRARY_PATH`:

```
export LD_LIBRARY_PATH=$(python3-config --prefix)/lib:$LD_LIBRARY_PATH
```

Cythonize:

```
cython -3 --cplus --embed tf_autodiff.pyx
```

Compile C++ script:
```
g++ tf_autodiff.cpp -o tf_autodiff $(python3-config --includes) -L$(python3-config --prefix)/lib -lpython3.7m
```
