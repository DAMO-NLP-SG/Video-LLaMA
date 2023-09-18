from setuptools import setup, find_packages


def _install_requirements():
    with open('requirement.txt') as f:
        packages = [line.strip() for line in f if not line.startswith('http')]
    return packages


setup(
    name='videollama',
    version='0.1.0',
    python_requires='>=3.8.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=_install_requirements(),
)
