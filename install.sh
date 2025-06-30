#!/bin/bash

echo "Instalare dependinte sistem"

# Update sistem
sudo apt update && sudo apt upgrade -y

# Instalare pachete necesare
sudo apt install -y python3-pip python3-dev libatlas-base-dev i2c-tools git

# Activare I2C (doar daca nu e activ)
sudo raspi-config nonint do_i2c 0

# Instalare pip dependencies
echo "Instalare dependinte Python"
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "Toate dependintele au fost instalate!"
