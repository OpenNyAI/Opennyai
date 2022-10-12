from setuptools import setup, find_packages

"""
Instructions for creating a release of the opennyai library.
1. Make sure your working directory is clean.
2. Make sure that you have changed the versions in "opennyai/__init__.py".
3. Create the distribution by running "python setup.py sdist" and " python setup.py bdist_wheel" in the root of the repository.
4. Check you can install the new distribution in a clean environment.
5. Upload the distribution to pypi by running
   "twine upload dist/*".
   This step will ask you for a username and password - the username & password get from [AUTHOR]
"""

setup(
    name="opennyai",
    version="0.0.4",
    url="",
    author="Aman Tiwari",
    author_email="aman.tiwari@thoughtworks.com",
    description="A SpaCy pipeline and models for NLP on indian legal text.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["law indianlegalner legalner lawtech legaltech nlp spacy SpaCy"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="MIT",
    install_requires=[
        "spacy==3.2.4",
        "spacy-transformers==1.1.5",
        "urllib3==1.26.11",
        "beautifulsoup4==4.10.0",
        "requests==2.28.1",
        "nltk==3.7",
        "tqdm>=4.63.0,<4.64.1"],
    tests_require=["pytest", "pytest-cov"],
    python_requires=">=3.7.0",
)
