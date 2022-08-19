#!/usr/bin/env bash
make clean build

./nppFilters ./data/CroppedYale/yaleB12/yaleB12_P00A+000E+00.pgm canny ./output/canny_test.bmp > output/output.log
./nppFilters ./data/CroppedYale/yaleB12/yaleB12_P00A+000E+00.pgm sobel ./output/sobel_test.bmp >> output/output.log
./nppFilters ./data/CroppedYale/yaleB12/yaleB12_P00A+000E+00.pgm gauss ./output/gauss_test.bmp >> output/output.log
./nppFilters ./data/CroppedYale/yaleB12/yaleB12_P00A+000E+00.pgm sharpen ./output/sharpen_test.bmp >> output/output.log
