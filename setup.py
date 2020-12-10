import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glam-browser-v2-cli-sminot",
    version="0.0.1",
    author="Samuel Minot",
    author_email="sminot@fredhutch.org",
    description="GLAM Browser CLI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FredHutch/glam-browser-v2",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.8',
    keywords='metagenomics microbiome science',
    install_requires=[
        "pandas>=0.20.3",
        "boto3>=1.4.7",
        "expiringdict",
        "fastcluster",
        "mysql",
        "mysql-connector-python-rf",
        "sqlalchemy",
        "h5py",
    ],
    entry_points={
        'console_scripts': [
            'glam-cli = helpers.cli:main',
        ],
    },
)
