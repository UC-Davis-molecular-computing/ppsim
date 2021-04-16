from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from os.path import join
from setuptools.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

inc_path = np.get_include()
lib_path = join(np.get_include(), '', '..', 'random', 'lib')

distributions = Extension("simulator",
                          sources=[join('', 'ppsim/simulator.pyx')],
                          include_dirs=[inc_path],
                          library_dirs=[lib_path],
                          libraries=['npyrandom']
                          )

setup(
    ext_modules=cythonize(distributions, compiler_directives={'language_level': "3"}, annotate=True)

)
