echo "Installing deps..."

# Install packages for running gym-retro
#apt-get update && apt-get install -y lua5.1 libav-tools

apt-get update && apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

pip install -e '.[all]'

echo "- Done!"