from distutils.core import setup
import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
	long_description = f.read()

setup(
  name="sightseer",
  version="1.0.5",
  description="State-of-the-art Computer Vision and Object Detection for TensorFlow.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url='https://github.com/rish-16/sight',
  download_url="https://github.com/rish-16/sight/archive/1.0.0.tar.gz",
  author="Rishabh Anand",
  author_email="mail.rishabh.anand@gmail.com",
  license="ASF",
  packages=["sightseer"],
  zip_safe=False
)