from setuptools import setup, find_packages

setup(
    name='RTransformer',
    version='1.0.0',
    license='none',
    description='RNN-Enhanced Transformer Module',

    author='Masao-Someki',
    url='https://github.com/Masao-Someki/RTransformer',

    packages=find_packages(where='utils'),
    package_dir={'': 'utils'},

    install_requires=[],
    extras_require={},

)
