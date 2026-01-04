from setuptools import setup, find_packages
import os

setup(
    name="a2agent",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10',
    author="Jialong Qin (jqinak@connect.ust.hk)",
    description="A Python package for A2AGENT",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)