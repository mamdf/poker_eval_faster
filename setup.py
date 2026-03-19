from pathlib import Path

from setuptools import Extension, find_packages, setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
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
    name="poker_eval_faster",
    version="1.0",
    description="High-performance poker hand evaluation for Texas Hold'em",
    long_description=README,
    long_description_content_type="text/markdown",
    author="marcos masci",
    python_requires=">=3.11",
    packages=find_packages(include=["poker_eval_faster", "poker_eval_faster.*"]),
    include_package_data=True,
    package_data={
        "poker_eval_faster": ["data/*.dat", "data/*.dat.gz", "data/*.bin"],
        "poker_eval_faster.eval_cython": ["*.pxd", "*.pyx", "*.c"],
    },
    ext_modules=build_extensions(),
    install_requires=[
        "click",
        "numpy",
    ],
    extras_require={
        "dev": [
            "build",
            "cython",
            "pytest",
            "pytest-benchmark",
        ],
    },
    entry_points={
        "console_scripts": [
            "poker-eval=poker_eval_faster.script:run",
        ]
    },
    zip_safe=False,
)
