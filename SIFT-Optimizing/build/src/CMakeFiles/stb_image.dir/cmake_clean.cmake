file(REMOVE_RECURSE
  "libstb_image.a"
  "libstb_image.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/stb_image.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
