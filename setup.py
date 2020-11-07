from setuptools import setup, find_packages

setup(
    name="inpaint",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="0.0.1",
    description="Generative model for painting masked images",
    # install_requires=open("requirements.txt").readlines(),
    setup_requires=["wheel"],
    author="tum.ai",
)
