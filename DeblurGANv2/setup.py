from setuptools import setup, find_packages

setup(
    name="deblurgan",
    version="0.1.0",
    description="DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better",
    author="Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang",
    author_email="",
    url="https://github.com/VITA-Group/DeblurGANv2",
    packages=find_packages(),
    package_data={
        'deblurgan': ['config/*.yaml'],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)