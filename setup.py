from setuptools import setup

setup(
    name='fastmrt',
    version='v0.1.0',
    packages=['fastmrt'],
    author='Sijie Xu',
    author_email='sijie.x@sjtu.edu.cn',
    description='A package for accerating MR thermometry by deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/minipuding/FastMRT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)