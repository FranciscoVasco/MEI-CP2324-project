#!/bin/bash

EXECUTABLE="../build/dataset_generator"
OUTPUT_DIR="../dataset"

IMAGES=(
  "$OUTPUT_DIR/image512.ppm 512 512"
  "$OUTPUT_DIR/image1024.ppm 1024 1024"
  "$OUTPUT_DIR/image2048.ppm 2048 2048"
  "$OUTPUT_DIR/image4096.ppm 4096 4096"
  "$OUTPUT_DIR/image8192.ppm 8192 8192"
  "$OUTPUT_DIR/image12288.ppm 12288 12288"
)

for IMAGE_PARAMS in "${IMAGES[@]}"; do
  echo "Generating image with parameters: $IMAGE_PARAMS"
  $EXECUTABLE $IMAGE_PARAMS
done

echo "Image generation done."