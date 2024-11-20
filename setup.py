from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

extensions = [
    Extension("dual_autodiff_x.dual", ["dual_autodiff_x/dual.pyx"]),
    Extension("dual_autodiff_x.autodiff_tools", ["dual_autodiff_x/autodiff_tools.pyx"]),
]

setup(
    name="dual_autodiff",  # Jacob make sure it the same as in pyproject.toml
    version="0.1.0",
    description="A Python package for forward-mode automatic differentiation using dual numbers with a Cythonized version for enhanced performance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
        "wheel",
        "Cython",
    ],
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
    python_requires=">=3.9",
)