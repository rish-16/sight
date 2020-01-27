import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
  name="sightseer",
  version="1.0.1",
  description="State-of-the-art Computer Vision and Object Detection for TensorFlow.",
  long_description=README,
  long_description_content_type="text/markdown",
  author="",
  author_email="",
  license="ASF",
  packages=["sightseer"],
  zip_safe=False
)