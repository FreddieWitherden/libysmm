
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was injaConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################
set(INJA_VERSION "3.3.0")

set(INJA_PACKAGE_USE_EMBEDDED_JSON "ON")

include(CMakeFindDependencyMacro)

if(NOT INJA_PACKAGE_USE_EMBEDDED_JSON)
    find_dependency(nlohmann_json REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/injaTargets.cmake")
