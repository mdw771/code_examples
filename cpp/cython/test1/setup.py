from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

cythonize('*.pyx', language='c++')
setup(ext_modules=[Extension('mod', sources=['mod.cpp', 'ops.cpp'], language='c++')])
