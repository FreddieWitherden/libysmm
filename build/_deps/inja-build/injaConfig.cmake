set(INJA_VERSION "3.3.0")

set(INJA_PACKAGE_USE_EMBEDDED_JSON "ON")

include(CMakeFindDependencyMacro)

if(NOT INJA_PACKAGE_USE_EMBEDDED_JSON)
    find_dependency(nlohmann_json REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/injaTargets.cmake")
