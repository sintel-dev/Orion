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
    'tensorflow>=2.16.1,<2.20',
    'numpy>=1.23.5,<2',
    'pandas>=1.4.0,<3',
    'numba>=0.56.2,<0.70',
    'mlblocks>=0.6.2,<0.8',
    'ml-stars>=0.2.2,<0.4',
    'scikit-learn>=1.1.0,<1.6',
    'scipy>=1.8.0,<2',
    'pyts>=0.11,<0.14',
    'torch>=1.12.0,<2.6',
    'azure-cognitiveservices-anomalydetector>=0.3,<0.4',
    'xlsxwriter>=1.3.6,<1.4',
    'tqdm>=4.36.1',
    'stumpy>=1.7,<1.11',
    'ncps',

    # fix conflict
    'protobuf<4',
]

pretrained_requires = [
    #units
    'timm',
    'smart_open',

    #timesfm
    "timesfm[torch]>=1.2.0,<1.5;python_version>='3.11' and python_version<'3.12'",
    "jax;python_version>='3.11' and python_version<'3.12'",

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
    'watchdog>=0.8.3,<5',

    # docs
    'docutils>=0.12,<1',
    'nbsphinx>=0.5.0,<1',
    'sphinx_toolbox>=2.5,<4',
    'Sphinx>=3,<8',
    'pydata-sphinx-theme<1',
    'markupsafe<3',
    'ipython>=6.5,<12',
    'Jinja2>=2,<4',
    'pickleshare', # ipython sphinx
    
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
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
        'pretrained': pretrained_requires,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='orion',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='orion-ml',
    packages=find_packages(include=['orion', 'orion.*']),
    python_requires='>=3.9,<3.13',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/sintel-dev/Orion',
    version='0.7.1.dev1',
    zip_safe=False,
)
