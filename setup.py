from setuptools import setup
from Cython.Build import cythonize

setup(
    name='poker_eval_faster',
    version='1.0',
    packages=['poker_eval_faster'],
    url='',
    license='',
    author='marcos masci',
    author_email='',
    description='evaluate poker hands',
    package_data={'poker_eval_faster': ['data/*']},
    ext_modules=cythonize('poker_eval_faster/eval_cython/*.pyx'),
    entry_points='''
        [console_scripts]
        poker-eval=poker_eval_faster.script:run
    ''',
    install_requires=['cython', 'numpy', 'future', 'pytest', 'click'],
)
