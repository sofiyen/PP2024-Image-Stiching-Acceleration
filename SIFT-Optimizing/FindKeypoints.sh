#!/bin/bash

# Compile the program
cd build && cmake -DCMAKE_CXX_COMPILER=hipcc -DOpenCV_DIR=/work1/ntucourse/ntustudent003/opencv/lib64/cmake/opencv4 .. && make

# Move to bin directory
cd ../bin/

# Remove old profile file if it exists and create a new empty one
# rm -f match_features_profile.txt
# touch match_features_profile.txt

# Loop through image pairs (1 through 4)
for i in {1..4}
do
    img="../imgs/images/${i}-1.jpg"
    
    # Check if both images exist
    if [ -f "$img" ]; then
        echo "Processing image ${i}: ${img}"
        # Run find_keypoints, grep for total time, and append to profile file
        echo "Image ${i}:" >> find_keypoints_profile.txt
        srun -p mi2104x -t 00:50:00 ./find_keypoints "$img" | grep "Total execution time:" >> find_keypoints_profile.txt
        echo "----------------------------------------" >> find_keypoints_profile.txt
    else
        echo "Warning: Image ${i} not found, skipping..."
    fi
done