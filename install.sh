echo "Installing deps..."

pip install gym

# Install packages for running gym-retro
#apt-get update && apt-get install -y lua5.1 libav-tools

#Ubuntu
apt-get update && apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

#OSX
# brew install cmake boost boost-python sdl2 swig wget

pip install -e '.[all]'

pip install 'gym[all]'

pip install box2d box2d-kengz

echo "- Done!"