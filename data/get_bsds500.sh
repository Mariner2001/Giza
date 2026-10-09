#!/bin/sh
# Downloads the 20 BSDS500 test images used in experiment E6 and their human segmentations
# into data/BSDS500/data/{images,groundTruth}/test (about 2.5 MB). Requires git >= 2.25.
# The BSDS500 data are provided by their authors for research use; they are not redistributed here.
set -e
cd "$(dirname "$0")"
IDS="118031 123057 16068 198087 235098 279005 285022 290035 317043 326085 335094 35028 365072 385022 41029 77062 8068 87015 92014 94095"
rm -rf bsds_tmp
git clone --depth 1 --filter=blob:none --sparse https://github.com/BIDS/BSDS500.git bsds_tmp
for i in $IDS; do echo "/BSDS500/data/images/test/$i.jpg"; echo "/BSDS500/data/groundTruth/test/$i.mat"; done > sparse.txt
(cd bsds_tmp && git sparse-checkout set --no-cone --stdin < ../sparse.txt)
rm -rf BSDS500 && mv bsds_tmp/BSDS500 BSDS500 && rm -rf bsds_tmp sparse.txt
echo "BSDS500 images in $(pwd)/BSDS500/data"
