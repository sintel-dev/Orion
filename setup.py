#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    with open('README.md', encoding='utf-8') as readme_file:
        readme = readme_file.read()
except IOError:
    readme = ''

try:
    with open('HISTORY.md', encoding='utf-8') as history_file:
        history = history_file.read()
except IOError:
    history = ''


install_requires = [
    'tensorflow>=2.2,<2.15',
    'numpy>=1.17.5,<2',
    'pandas>=1,<2',
    'numba>=0.48,<0.60',
    's3fs>=0.2.2,<0.5',
    'mlblocks>=0.6.1,<0.7',
    'ml-stars>=0.2,<0.3',
    'scikit-learn>=0.22.1,<1.2',
    'scipy<1.14',
    'tabulate>=0.8.3,<0.9',
    'pyts>=0.11,<0.14',
    'torch>=1.4',
    'azure-cognitiveservices-anomalydetector>=0.3,<0.4',
    'xlsxwriter>=1.3.6,<1.4',
    'tqdm>=4.36.1',
    'stumpy>=1.7,<1.11',
    'ncps',

    # fix conflict
    'protobuf<4',
    
    # fails on python 3.6
    'opencv-python<4.7',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
    'rundoc>=0.4.3,<0.5',
]

development_requires = [
    # general
    'pip>=9.0.1',
    'bumpversion>=0.5.3,<0.6',
    'watchdog>=0.8.3,<0.11',

    # docs
    'docutils>=0.12,<0.18',
    'm2r2>=0.2.5,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=3,<3.3',
    'pydata-sphinx-theme<0.5',
    'markupsafe<2.1.0',
    'ipython>=6.5,<9',
    'Jinja2>=2,<3',

    # fails on Sphinx < v3.4
    'alabaster<=0.7.12',
    # fails on Sphins < v5.0
    'sphinxcontrib-applehelp<1.0.8',
    'sphinxcontrib-devhelp<1.0.6',
    'sphinxcontrib-htmlhelp<2.0.5',
    'sphinxcontrib-serializinghtml<1.1.10',
    'sphinxcontrib-qthelp<1.0.7',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.2,<2',
    'autopep8>=1.4.3,<2',
    'importlib-metadata<5',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',
    'invoke',
]

setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="Orion is a machine learning library built for unsupervised time series anomaly detection.",
    entry_points={
        'console_scripts': [
            'orion=orion.__main__:main'
        ],
        'mlblocks': [
            'primitives=orion:MLBLOCKS_PRIMITIVES',
            'pipelines=orion:MLBLOCKS_PIPELINES'
        ],
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='orion',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='orion-ml',
    packages=find_packages(include=['orion', 'orion.*']),
    python_requires='>=3.8,<3.12',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sintel-dev/Orion',
    version='0.6.1.dev0',
    zip_safe=False,
)
