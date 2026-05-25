"""Setup script for acx-ramanujan PyPI package."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='acx-ramanujan',
    version='2.0.0',
    author='aaroncxxx',
    author_email='122241711@qq.com',
    description='Ramanujan modular function recurrence for neural network weight initialization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aaroncxxx/ACX-ramanujan-transformer',
    project_urls={
        'Bug Reports': 'https://github.com/aaroncxxx/ACX-ramanujan-transformer/issues',
        'Source': 'https://github.com/aaroncxxx/ACX-ramanujan-transformer',
        'Documentation': 'https://github.com/aaroncxxx/ACX-ramanujan-transformer/blob/main/docs/theory.md',
    },
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
    extras_require={
        'dev': ['pytest>=7.0', 'black', 'flake8'],
        'hf': ['transformers>=4.30.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords=[
        'transformer', 'initialization', 'ramanujan', 'modular-forms',
        'neural-network', 'deep-learning', 'weight-initialization',
        'variance-preservation', 'mixture-of-experts',
    ],
)
