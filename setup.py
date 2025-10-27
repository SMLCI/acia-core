#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="utf-8") as history_file:
    history = history_file.read()


with open("requirements.txt", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Johannes Seiffarth",
    author_email="j.seiffarth@fz-juelich.de",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="The acia library provides utility functionality for analyzing 2D+t time-lapse image sequences in microfluidic live-cell imaging experiments.",
    entry_points={
        "console_scripts": [
            "acia=acia.cli:main",
        ],
    },
    install_requires=requirements,
    extras_require={
        "omero": ["omero-py>=5.9.3"],
    },
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="acia",
    name="acia",
    packages=find_packages(include=["acia", "acia.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/JojoDevel/acia",
    version="0.3.2",
    zip_safe=False,
)
