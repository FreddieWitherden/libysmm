cmake_minimum_required(VERSION 3.9)

set(LIBYSMM_VERSION_MAJOR 0)
set(LIBYSMM_VERSION_MINOR 1)
set(LIBYSMM_VERSION_PATCH 0)
set(LIBYSMM_VERSION "${LIBYSMM_VERSION_MAJOR}.${LIBYSMM_VERSION_MINOR}.${LIBYSMM_VERSION_PATCH}")

project(libysmm VERSION ${LIBYSMM_VERSION} DESCRIPTION "libysmm")

add_library(ysmm-cl SHARED
    src/cl.cpp
)

find_package(OpenCL REQUIRED)

set_target_properties(ysmm-cl PROPERTIES VERSION ${PROJECT_VERSION})
target_compile_features(ysmm-cl PRIVATE cxx_std_17)
target_include_directories(ysmm-cl PUBLIC include)
target_include_directories(ysmm-cl PRIVATE src)
target_include_directories(ysmm-cl PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_directories(ysmm-cl PRIVATE ${OpenCL_LIBRARY})
target_link_libraries(ysmm-cl OpenCL)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.cmake
               ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h)

install(TARGETS ysmm-cl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
