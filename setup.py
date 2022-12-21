from setuptools import setup, find_packages

"""
Instructions for creating a release of the opennyai library.
1. Make sure your working directory is clean.
2. Make sure that you have changed the versions in "opennyai/__init__.py".
3. Create the distribution by running "python setup.py sdist" and "python setup.py bdist_wheel" in the root of the repository.
4. Check you can install the new distribution in a clean environment.
5. Upload the distribution to pypi by running
   "twine upload dist/*".
   This step will ask you for a username and password - the username & password get from [AUTHOR]
"""

setup(
    name="opennyai",
    version="0.0.10",
    url="",
    author="Aman Tiwari",
    author_email="aman.tiwari@thoughtworks.com",
    description="A SpaCy pipeline and models for NLP on indian legal text.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=[
        "law indianlegalner legalner legal ner lawtech legaltech nlp spacy SpaCy rhetorical role summarizer extractive_summarizer "],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="MIT",
    install_requires=[
        "torch<1.13.0,>=1.10.0",
        "torchvision<0.14.0,>=0.8.1",
        "transformers<4.16,>=4.1",
        "pytorch-transformers==1.2.0",
        "multiprocess==0.70.12.2",
        "pandas>=1.2.4,<1.3.6",
        "spacy<3.2.5,>=3.2.2",
        "spacy-transformers<1.1.6,>=1.1.4",
        "urllib3<1.26.12,>=1.26.8",
        "beautifulsoup4<4.11.0,>=4.10.0",
        "requests<2.28.2,>=2.27.1",
        "nltk<3.6.6,>=3.6",
        "tqdm>=4.63.0,<4.64.1",
        "prettytable>=3.1.1,<3.4.1",
        "scikit-learn", "pytest",
        "Levenshtein"],
    tests_require=["pytest", "pytest-cov"],
    python_requires=">=3.7.0",
)
