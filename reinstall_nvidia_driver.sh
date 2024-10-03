#!/bin/bash

# Script to reinstall NVIDIA driver on Debian systems
# Usage: sudo ./reinstall_nvidia_driver.sh

set -e

echo "Starting NVIDIA driver reinstallation script..."

# Step 1: Update the system package lists
echo "Updating package lists..."
sudo apt update

# Step 2: Remove existing NVIDIA drivers
echo "Removing existing NVIDIA drivers..."
sudo apt remove --purge '^nvidia-.*' -y
sudo apt autoremove -y

# Step 3: Install required dependencies
echo "Installing required dependencies..."
sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# Step 4: Enable contrib and non-free repositories
echo "Enabling contrib and non-free repositories..."
sudo sed -i '/deb.*main/s/$/ contrib non-free non-free-firmware/' /etc/apt/sources.list

# Step 5: Update package lists after enabling new repositories
echo "Updating package lists after adding contrib and non-free..."
sudo apt update

# Step 6: Install the NVIDIA driver and firmware
echo "Installing NVIDIA driver and firmware..."
sudo apt install -y nvidia-driver firmware-misc-nonfree

# Step 7: Blacklist the Nouveau driver
echo "Blacklisting the Nouveau driver..."
sudo bash -c 'echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist-nouveau.conf'

# Step 8: Update initramfs
echo "Updating initramfs..."
sudo update-initramfs -u

echo "NVIDIA driver reinstallation complete."

echo "Please reboot your system to apply the changes."