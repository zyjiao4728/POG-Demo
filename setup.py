# python setup.py bdist_wheel
# sudo pip install ./dist/*.whl

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    README = fh.read()

requirements = [
    'pyparsing>=2.4.7',
    'python-fcl>=0.6.1',
    'matplotlib>=3.4.2',
    'networkx>=2.6.2',
    'numpy>=1.21.1',
    'transforms3d>=0.3.1',
    'pybullet>=3.2.0',
    'trimesh>=3.9.27',
    'vedo>=2021.0.7'
]

setup(
    name="pog",
    version="0.0.1",

    author="Ziyuan Jiao",
    author_email="zyjiao@ucla.edu",
    
    description="Planning on Scene Graph",
    long_description=README,
        
    packages=['pog'],
    include_package_data=True,

    license="Apache-2.0",

    test_suite='tests',

    install_requires=requirements,

    python_requires='>=3.6, <4',
    
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ),
)