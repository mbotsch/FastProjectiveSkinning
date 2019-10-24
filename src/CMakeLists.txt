file(GLOB_RECURSE SOURCES *.cpp)
file(GLOB_RECURSE HEADERS *.h)
file(GLOB_RECURSE KERNELS *.cu)

#todo: are opengl_libraries required?

if(CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(skinning ${HEADERS} ${SOURCES} ${KERNELS})
    target_link_libraries(skinning pmp_vis ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})
else()
    add_executable(skinning ${HEADERS} ${SOURCES})
    target_link_libraries(skinning pmp_vis ${OPENGL_LIBRARIES})
endif()
