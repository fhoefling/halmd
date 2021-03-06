# find Lua source files in source tree
file(GLOB_RECURSE halmd_lua_sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "halmd/*.lua.in")

# skip disabled potential modules
foreach(file ${halmd_lua_sources})
  if(${file} MATCHES "halmd/mdsim/potentials/(.*)/(.*)\\.lua\\.in")
    if(NOT HALMD_WITH_${CMAKE_MATCH_1}_${CMAKE_MATCH_2} AND NOT ${CMAKE_MATCH_2} STREQUAL "init"
       AND NOT ${CMAKE_MATCH_1} MATCHES "/adapters$")
      list(REMOVE_ITEM halmd_lua_sources ${file})
    endif()
  endif()
endforeach()

if(HALMD_VARIANT_HOST_SINGLE_PRECISION)
  set(HALMD_HOST_FLOAT_TYPE "float")
  set(HALMD_HOST_PRECISION "single")
else()
  set(HALMD_HOST_FLOAT_TYPE "double")
  set(HALMD_HOST_PRECISION "double")
endif()

if(HALMD_VARIANT_GPU_DOUBLE_SINGLE_PRECISION)
  set(HALMD_DEFAULT_GPU_PRECISION "double-single")
else()
  set(HALMD_DEFAULT_GPU_PRECISION "single")
endif()


# copy files from source to build tree
foreach(file ${halmd_lua_sources})
  string(REGEX REPLACE "\\.in$" "" out_file ${file})
  configure_file(${file} ${out_file} @ONLY)
endforeach()

# install Lua files from build tree
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/halmd"
  DESTINATION share/halmd/lua
  FILES_MATCHING PATTERN "*.lua"
)
