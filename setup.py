from setuptools import Extension, setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension("PyTwoPlusTwoEval", ["twoplustwo_eval/evaluate.pyx"], include_dirs=['.']),
]

setup(
    name='PyTwoPlusTwoEval',
    version='1.0',
    packages=['twoplustwo_eval'],
    url='',
    license='',
    author='marcos',
    author_email='',
    description='',
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext': {'inplace': True, 'force': False}}
)
