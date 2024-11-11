import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="circular_clustering",
    version="0.0.25",
    author="Hugo J. Bello",
    author_email="hjbello.wk@protonmail.com",
    description="Adaptation of X means algorithm for circular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/General-Clustering/circular_clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)