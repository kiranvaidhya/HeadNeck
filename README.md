# Mandible Segmentation from Head and Neck CT Volumes

## Data
Download the head and neck data from the [MICCAI Challenge](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge) site.

### Data preparation

Data split has been listed in `dataList.ods`.

Patch and Slice extraction codes inside HeadNeck/codes

## Training

Trains a patch based CNN to classify whether the given patch belongs to Mandible or not. Patch size of `21x21` was chosen.

### TODO:
Convert the patch based network into a fully convolutional network.

## Results

The best network was tested on Validation+Testing Dataset and the dice similarity score was `0.53`.
