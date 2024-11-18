from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "dual_autodiff_x.dual", 
        ["dual_autodiff_x/dual.pyx"]  
    ),
    Extension(
        "dual_autodiff_x.autodiff_tools", 
        ["dual_autodiff_x/autodiff_tools.pyx"]  
    ),
]

setup(
    name="dual_autodiff_x",
    version="0.1.0",
    description="A Cythonized version of the dual_autodiff package for enhanced performance",
    author="Jacob Tutt",
    author_email="jacobtutt@icloud.com",
    url="https://github.com/JacobTutt/dual_autodiff_x", 
    packages=["dual_autodiff_x"],  
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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  
)