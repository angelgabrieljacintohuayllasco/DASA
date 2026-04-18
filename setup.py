from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dasa",
    version="0.1.0",
    description="Deterministic Agent Synthesis Architecture — hallucination-free AI on minimal hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="DASA Contributors",
    license="Apache-2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    install_requires=[
        "sentence-transformers>=2.6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
