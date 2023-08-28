#!/bin/bash

for file in ILSVRC2012_val_*; do
    # Split the filename into an array using the underscore as a delimiter
    IFS='_' read -ra parts <<< "$file"
    # Construct the new filename by removing the third element of the array
    newfile="${parts[0]}_${parts[1]}_${parts[2]}.JPEG"
    # Rename the file
    mv "$file" "$newfile"
done
