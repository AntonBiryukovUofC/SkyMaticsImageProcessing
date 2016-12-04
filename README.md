# SkyMaticsImageProcessing
The repo for my work with Skymatics.

October 1, Sat: Tile the large (1 GB+) tiff images into smaller images using VIPS library and its scripts.
As an alternative, GDAL library seems to be a good candidate, as it 's done specifically for georeferenced TIFFs.

October : Mastered tiling using gdal2tiles.py, the instructions to follow soon.
Added the unsupervised clustering of images based on the histogram, fractal dimensions and Gabor-filtered images.

Nov 5:
We need to start a Google Drive account / Dropbox account so we could also store the images for each classifier, without rebuilding them all the time.
With the next commit I will add the comments into each file and explain what the file purpose is.

# Fix the gdal2tiles EPSG error
The fix is to change the following line in gdal2tile.py (line 785):

if self.options.profile == 'mercator':
    self.out_srs.ImportFromEPSG(900913)

to:

if self.options.profile == 'mercator':
    self.out_srs.ImportFromEPSG(3857)

This one change allows gdal2tiles.py to run without error.

# A note on the feature extraction:

I give a bit of thought towards using the layers of the pre-trained CNNs to help me construct the features for the classification. Right now it seems a bit complicated + I would like to engineer features myself first to learn what is important and what is not in the analysis. Therefore, I ll stick to the features I used previously.

Now, here's a few things I'd like to test.

a) Resize the tiles from 256x256 to 64x64. This will positively affect the processing time for the Gabor image calculation, as well as reduce the memory used. Anticipating at least four-fold increase in the processing time.
b) Try to use the ratios of the sum of Canny -Edge images with two different values of sigma. This will give me an idea of how heavy the tails of the distribution are. With the tiles including animal tracks, this ratio would definitely be larger than in those that are flat





Stay tuned!
