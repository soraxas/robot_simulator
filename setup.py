import re
from os import path

from setuptools import setup

# read the contents of README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version_str = "v0.1.0"

setup(
    name="robosim",
    version=version_str,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tin Lai (@soraxas)",
    author_email="oscar@tinyiu.com",
    license="MIT",
    url="https://github.com/soraxas/robot_simulator",
    keywords="",
    python_requires=">=3.6",
    packages=[
        "robosim",
        "robosim.scene",
        "robosim.learning",
        "robosim.simulator",
        "robosim.trajectory_demonstration",
    ],
    install_requires=[
        "torch",
        "differentiable-robot-model @ git+https://github.com/soraxas/differentiable-robot-model.git@feature-various-stuff#egg=differentiable-robot-model",
        "pybullet-planning @ git+https://github.com/soraxas/pybullet-planning@f580bad01479d657b9ee549b440e33706b10318d",
        "numpy-quaternion",
        "scipy",
    ],
    classifiers=[
        "Environment :: Console",
        "Framework :: Matplotlib",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Desktop Environment",
    ],
)
