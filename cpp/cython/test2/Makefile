CC=x86_64-conda-linux-gnu-g++

LIB_PYTHON=$(shell python3-config --cflags --libs)

all: pythonLib cppExecutable

pythonLib:
	python setup.py  build_ext --inplace

cppExecutable:
	$(CC) $(LIB_PYTHON) -I./py_c.cpython-39-x86_64-linux-gnu.so run.c -o run

clean:
	$(RM) py_c.c *.so *.h

