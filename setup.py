from pathlib import Path

from setuptools import Distribution, setup


class BinaryAwareDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        package_dir = Path(__file__).parent / "src" / "dmi"
        return any(package_dir.glob("_native_backend*.so"))


setup(distclass=BinaryAwareDistribution)
