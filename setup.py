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
  url='https://github.com/rish-16/sight',
  download_url="https://github.com/rish-16/sight/archive/1.0.0.tar.gz",
  author="Rishabh Anand",
  author_email="mail.rishabh.anand@gmail.com",
  license="ASF",
  packages=["sightseer"],
  zip_safe=False
)