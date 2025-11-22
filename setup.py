from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pansharpening_metrics",
    version="1.0.0",
    author="Riccardo Musto",
    description="Quality metrics for hyperspectral pansharpening evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pansharpening-metrics", # <--- DA CAMBIARE
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-image>=0.17.0",
        "dask[distributed]>=2021.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "geospatial": ["rasterio>=1.2.0"],
    },
)