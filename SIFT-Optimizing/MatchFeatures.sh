#!/bin/bash

# Compile the program
cd build && cmake -DCMAKE_CXX_COMPILER=hipcc .. && make

# Move to bin directory
cd ../bin/

# Remove old profile file if it exists and create a new empty one
# rm -f match_features_profile.txt
# touch match_features_profile.txt

# Loop through image pairs (1 through 4)
for i in {1..4}
do
    img1="../imgs/images/${i}-1.jpg"
    img2="../imgs/images/${i}-2.jpg"
    
    # Check if both images exist
    if [ -f "$img1" ] && [ -f "$img2" ]; then
        echo "Processing image pair ${i}: ${img1} ${img2}"
        # Run match_features, grep for total time, and append to profile file
        echo "Image pair ${i}:" >> match_features_profile.txt
        srun -p mi2104x -t 00:50:00 ./match_features "$img1" "$img2" | grep "Total executing time:" >> match_features_profile.txt
        echo "----------------------------------------" >> match_features_profile.txt
    else
        echo "Warning: Image pair ${i} not found, skipping..."
    fi
done