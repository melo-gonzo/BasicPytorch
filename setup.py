from distutils.core import setup

setup(
    name="basic_pytorch",
    version="0.1",
    packages=[
        "data-modules",
        "models",
        "pipelines",
        "configs",
    ],
    long_description=open("README.md").read(),
)
