add_library(
    dx2
    empty.cxx
)

add_executable(predict
            predict.cxx)

target_link_libraries(predict
    PRIVATE
    dx2
) 

target_link_libraries(dx2
    PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    hdf5::hdf5
    gemmi::gemmi_cpp
    argparse
    fmt
)

target_include_directories(
    dx2 PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${mdspan_SOURCE_DIR}/include>
)
