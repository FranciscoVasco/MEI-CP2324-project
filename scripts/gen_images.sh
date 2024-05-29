#!/bin/bash

EXECUTABLE="./dataset_generator"
OUTPUT_DIR="./dataset"

IMAGES=(
  "image512 $OUTPUT_DIR/image512.ppm 512 512"
  "image1024 $OUTPUT_DIR/image1024.ppm 1024 1024"
  "image2048 $OUTPUT_DIR/image2048.ppm 2048 2048"
  "image4096 $OUTPUT_DIR/image4096.ppm 4096 4096"
)

for IMAGE_PARAMS in "${IMAGES[@]}"; do
  echo "Generating image with parameters: $IMAGE_PARAMS"
  $EXECUTABLE $IMAGE_PARAMS
done

echo "Image generation done."