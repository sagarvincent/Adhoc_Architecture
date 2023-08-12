# Adhoc_Architecture


sudo apt install git
git clone https://github.com/LARG/HFO.git
sudo apt-get install libboost-all-dev
sudo add-apt-repository ppa:rock-core/qt4
sudo apt update
sudo apt install qt4-dev-tools libqt4-dev libqtcore4 libqtgui4
sudo apt install pip
pip install numpy
sudo apt install flex
sudo apt install python-is-python3

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelwithDebInfo ..
make -j4
make install


pip install scikit-learn
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
export JAVA_HOME=/usr/java/latest
sudo apt install default-jdk
pip install scikit-spatial
pip install sklearn-weka-plugin



