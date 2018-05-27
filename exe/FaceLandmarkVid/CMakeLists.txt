#TBB library
include_directories(${TBB_ROOT_DIR}/include)
include_directories(${BLAS_ROOT_DIR})

#OpenBlas library
include_directories(${BLAS_ROOT_DIR})
include_directories(../../lib/3rdParty/OpenBLAS/include)

add_executable(FaceLandmarkVid FaceLandmarkVid.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)
include_directories(../../lib/local/GazeAnalyser/include)
include_directories(../../lib/local/Utilities/include)		

target_link_libraries(FaceLandmarkVid LandmarkDetector)
target_link_libraries(FaceLandmarkVid FaceAnalyser)
target_link_libraries(FaceLandmarkVid GazeAnalyser)
target_link_libraries(FaceLandmarkVid Utilities)

target_link_libraries(FaceLandmarkVid dlib)

target_link_libraries(FaceLandmarkVid ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})

install (TARGETS FaceLandmarkVid DESTINATION bin)
