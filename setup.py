from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
from os.path import join
from setuptools.extension import Extension


inc_path = np.get_include()
lib_path = join(np.get_include(), '..', '..', 'random', 'lib')

with open("README.md", 'r') as f:
    long_description = f.read()

distributions = Extension("ppsim.simulator",
                          sources=[join('', 'ppsim/simulator.pyx')],
                          include_dirs=[inc_path],
                          library_dirs=[lib_path],
                          libraries=['npyrandom']
                          )

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    name="ppsim",
    packages=['ppsim'],
    version="0.0.7",
    author="Eric Severson",
    description="A package for simulating population protocols.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/UC-Davis-molecular-computing/population-protocols-python-package",
    ext_modules=cythonize(distributions, compiler_directives={'language_level': "3"}),
    install_requires=install_requires
)
