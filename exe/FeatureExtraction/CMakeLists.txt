#TBB library
include_directories(${TBB_ROOT_DIR}/include)

#OpenBlas library
include_directories(${BLAS_ROOT_DIR})
include_directories(../../lib/3rdParty/OpenBLAS/include)

add_executable(FeatureExtraction FeatureExtraction.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)		
include_directories(../../lib/local/GazeAnalyser/include)		
include_directories(../../lib/local/Utilities/include)		

target_link_libraries(FeatureExtraction LandmarkDetector)
target_link_libraries(FeatureExtraction FaceAnalyser)
target_link_libraries(FeatureExtraction GazeAnalyser)
target_link_libraries(FeatureExtraction Utilities)
target_link_libraries(FeatureExtraction dlib)

target_link_libraries(FeatureExtraction ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(FeatureExtraction ${DLIB_LIBRARY})

install (TARGETS FeatureExtraction DESTINATION bin)
