from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'
def get_requirements(file_path:str)-> list[str]:
    """
    this function will return the list of requirement """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Abdullah',
    author_email='abdullahmansoor123@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')


)