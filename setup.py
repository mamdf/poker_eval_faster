from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


ROOT = Path(__file__).parent
EXTENSION_NAMES = (
    "common",
    "hands_evaluate",
    "main",
    "one_hand_evaluate",
    "python_wrapper",
)
USE_CYTHON = cythonize is not None
SOURCE_SUFFIX = ".pyx" if USE_CYTHON else ".c"


def build_extensions() -> list[Extension]:
    extensions = [
        Extension(
            name=f"poker_eval_faster.eval_cython.{extension_name}",
            sources=[f"poker_eval_faster/eval_cython/{extension_name}{SOURCE_SUFFIX}"],
        )
        for extension_name in EXTENSION_NAMES
    ]
    if not USE_CYTHON:
        return extensions
    return cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )


setup(
    ext_modules=build_extensions(),
)
