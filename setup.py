from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("dual_autodiff_x.dual", ["dual_autodiff_x/dual.pyx"]),
    Extension("dual_autodiff_x.autodiff_tools", ["dual_autodiff_x/autodiff_tools.pyx"]),
]

setup(
    name="dual_autodiff",  # Make it the same as in pyproject.toml
    version="0.1.0",
    description="A Python package for forward-mode automatic differentiation using dual numbers with a Cythonized version for enhanced performance",
    author="Jacob Tutt",
    author_email="jacobtutt@icloud.com",
    url="https://github.com/JacobTutt/dual_autodiff",
    packages=find_packages(where=["dual_autodiff", "dual_autodiff_x"]),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
    install_requires=[],
    setup_requires=[
        "setuptools",
        "Cython",
    ],
    extras_require={
        "testing": ["pytest"],
        "docs": ["sphinx", "nbsphinx", "jupyter"],
        "cython": ["cython"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)