#TBB library
include_directories(${TBB_ROOT_DIR}/include)

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/GazeEstimation.cpp
)

SET(HEADERS
    include/GazeEstimation.h
)

include_directories(./include)
include_directories(${GazeAnalyser_SOURCE_DIR}/include)

add_library( GazeAnalyser ${SOURCE} ${HEADERS} )

install (TARGETS GazeAnalyser DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
