from setuptools import setup
from Cython.Build import cythonize

setup(
    name='twoplustwo_eval',
    version='1.0',
    packages=['twoplustwo_eval'],
    url='',
    license='',
    author='marcos masci',
    author_email='',
    description='evaluate poker hands',
    ext_modules=cythonize('twoplustwo_eval/eval_cython/*.pyx'),
    entry_points='''
        [console_scripts]
        poker-eval=twoplustwo_eval.script:run
    ''',
    install_requires=['cython', 'numpy', 'future', 'pytest', 'click'],
)
