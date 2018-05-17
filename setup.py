from setuptools import setup, find_packages

install_requires = [
    'sphinx',
    'sphinx_gallery',
]

setup(
    name='dask-ml-benchmarks',
    description='A library for distributed and parallel machine learning',
    url='https://github.com/dask/dask-ml-benchmarks',

    author='Tom Augspurger',
    author_email='taugspurger@anaconda.com',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Database',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['docs', 'tests', 'tests.*', 'docs.*']),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=install_requires,
)
