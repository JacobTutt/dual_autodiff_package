from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import shutil
from pathlib import Path

# Function to copy all .py files from a source directory to the cython directory
# Exclude the __init.__.py and version.py files
def prepare_cython_sources(source_dir, cython_dir):
    Path(cython_dir).mkdir(parents=True, exist_ok=True)

    for py_file in Path(source_dir).rglob("*.py"):
        relative_path = py_file.relative_to(source_dir)

        if relative_path.name in ["__init__.py", "version.py"]:
            continue

        # Rename .py to .pyx and place in the Cython directory
        cython_file = Path(cython_dir) / relative_path.with_suffix(".pyx")
        cython_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the subdirectories exist
        shutil.copy(py_file, cython_file)

# execute the copy from dual_autodiff to dual_autodiff_x
prepare_cython_sources('dual_autodiff', 'dual_autodiff_x')

# Defines the Cython extensions dynamically
extensions = [
    Extension(
        name=f"dual_autodiff_x.{pyx_file.stem}",
        sources=[str(pyx_file)],
    )
    for pyx_file in Path("dual_autodiff_x").rglob("*.pyx")
]

setup(
    name="dual_autodiff", 
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
    install_requires=["numpy"],
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