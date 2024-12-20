# Define executable for find_keypoints
add_executable(find_keypoints find_keypoints.cpp)

# Include directories
# ../src contains custom headers, add OpenCV_INCLUDE_DIRS for OpenCV headers
target_include_directories(find_keypoints PRIVATE 
    ../src
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
# img and sift are custom libraries, OpenCV_LIBS links necessary OpenCV libraries
target_link_libraries(find_keypoints PRIVATE
    img
    sift
    ${OpenCV_LIBS}
    stdc++
)

# Define executable for match_features
add_executable(match_features match_features.cpp)

# Include directories
target_include_directories(match_features PRIVATE 
    ../src
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(match_features PRIVATE
    img
    sift
    ${OpenCV_LIBS}
    stdc++
)


# Define executable for match_features
add_executable(image_stitching image_stitching.cpp)

# Include directories
target_include_directories(image_stitching PRIVATE 
    ../src
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(image_stitching PRIVATE
    img
    sift
    ${OpenCV_LIBS}
    stdc++
)

# Set output directories for executables
set_target_properties(find_keypoints PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(match_features PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set_target_properties(image_stitching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)