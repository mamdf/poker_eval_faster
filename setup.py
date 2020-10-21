from setuptools import setup
from Cython.Build import cythonize

setup(
    name='twoplustwo_eval',
    version='1.0',
    packages=['twoplustwo_eval'],
    url='',
    license='',
    author='marcos',
    author_email='',
    description='evaluate poker hands',
    ext_modules=cythonize('twoplustwo_eval/*.pyx'),
    install_requires=['cython', 'numpy', 'future', 'pytest', 'scipy'],
)
