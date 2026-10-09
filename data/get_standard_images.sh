#!/bin/sh
# Downloads the colour Baboon (Mandrill) and Peppers test images used in Sect. 3.4 (audit_seg.py)
# into data/standard_test_images.
set -e
cd "$(dirname "$0")"
rm -rf sti_tmp
git clone --depth 1 https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing.git sti_tmp
mkdir -p standard_test_images
cp sti_tmp/standard_test_images/baboon.png sti_tmp/standard_test_images/peppers.png standard_test_images/
rm -rf sti_tmp
echo "images in $(pwd)/standard_test_images"
