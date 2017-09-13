#!/usr/bin/env bash

conda install -y scikit-learn
conda install -y -c conda-forge librosa

sudo apt-get install openjdk-8-jdk -y --force-yes -qq
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer -y --force-yes -qq

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install bazel -y --force-yes -qq

cd ~
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 -k //tensorflow/tools/pip_package:build_pip_package
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.3.0-py2-none-any.whl
