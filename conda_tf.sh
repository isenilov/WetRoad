#!/usr/bin/env bash

# installing Anaconda packages
conda install -y scikit-learn keras
conda install -y -c conda-forge librosa

# preparing env for bazel
sudo apt-get install openjdk-8-jdk -y --force-yes -qq
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer -y --force-yes -qq

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install bazel -y --force-yes -qq

# building tensorflow
cd ~
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
./configure
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 -k //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
