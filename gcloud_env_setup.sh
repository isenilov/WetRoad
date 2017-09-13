#!/usr/bin/env bash
# setup of environmet on Google Cloud VM instance

export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update -y --force-yes -qq
sudo apt-get install gcsfuse -y --force-yes -qq

conda install -y scikit-learn
conda install -y -c conda-forge librosa tensorflow tensorboard

cd WetRoad
mkdir models
mkdir dataset
gcsfuse --implicit-dirs wetroad_dataset ~/WetRoad/dataset

echo '# mounting of dataset bucket' >> ~/.bashrc
echo 'gcsfuse --implicit-dirs wetroad_dataset ~/WetRoad/dataset' >> ~/.bashrc