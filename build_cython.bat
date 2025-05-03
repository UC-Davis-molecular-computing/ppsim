@echo off

echo Building Cython extension...
python setup.py build_ext --inplace
