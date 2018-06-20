cmake_minimum_required (VERSION 2.8)
project (OpenFace)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(CMAKE_CONFIG_DIR etc/OpenFace)
set(CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_CONFIG_DIR}")
add_definitions(-DCONFIG_DIR="${CONFIG_DIR}")

find_package( BLAS REQUIRED )
include_directories( ${BLAS_INCLUDE_DIRS} )
LINK_DIRECTORIES(${BLAS_LIBRARY_DIRS})

find_package( OpenCV 3.3 REQUIRED )

MESSAGE("OpenCV information:") 
MESSAGE("  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}") 
MESSAGE("  OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}") 
MESSAGE("  OpenCV_LIBRARY_DIRS: ${OpenCV_LIBRARY_DIRS}") 
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( Boost 1.5.9 REQUIRED COMPONENTS filesystem system)
MESSAGE("Boost information:") 
MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}") 
MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}/boost)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})	

find_package(TBB REQUIRED)

# Move LandmarkDetector model
file(GLOB files "lib/local/LandmarkDetector/model/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

# Move the hierarchical LandmarkDetector models
file(GLOB files "lib/local/LandmarkDetector/model/model*")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	endif(MSVC)

	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/detection_validation/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/detection_validation)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/detection_validation)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/detection_validation)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/detection_validation)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/patch_experts/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/patch_experts)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/patch_experts)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/patch_experts/*.dat")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/patch_experts)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/patch_experts)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/mtcnn_detector/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/mtcnn_detector)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/mtcnn_detector)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/mtcnn_detector/*.dat")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/mtcnn_detector)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/mtcnn_detector)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

file(GLOB files "lib/local/LandmarkDetector/model/pdms/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/model/pdms)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/model/pdms)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/pdms)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/pdms)
endforeach()

# Move OpenCV classifiers
file(GLOB files "lib/3rdParty/OpenCV3.1/classifiers/*.xml")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/classifiers)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/classifiers)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/classifiers)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/classifiers)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/*.txt")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/AU_predictors)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/AU_predictors)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	endif(MSVC)

	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/svr*")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/AU_predictors)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/AU_predictors)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	endif(MSVC)

	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/svm*")
foreach(file ${files})
	if (MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug/AU_predictors)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release/AU_predictors)
	else(MSVC)
		file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	endif(MSVC)

	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.7)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -msse -msse2 -msse3")
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -msse3")
    endif ()
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -msse3")
endif ()

# Boost
if(WIN32)
	include_directories(lib/3rdParty/boost)
	include_directories(lib/3rdParty/boost/boost)
	link_directories( ${PROJECT_SOURCE_DIR}/lib/3rdParty/boost/lib )
else()
	INCLUDE_DIRECTORIES(${BOOST_INCLUDE_DIR})
endif()

# dlib library, allow specifying own or use the included one
if(NOT DEFINED ${DLIB_INCLUDE_DIR})
	set(DLIB_INCLUDE_DIR lib/3rdParty/dlib)
	set(DLIB_LIBRARY dlib.lib)
	add_subdirectory(lib/3rdParty/dlib)
endif()

MESSAGE("DLIB information:") 
MESSAGE("  DLIB_INCLUDE_DIR: ${DLIB_INCLUDE_DIR}") 
MESSAGE("  DLIB libraries: ${DLIB_LIBRARY}") 

# LandmarkDetector library
add_subdirectory(lib/local/LandmarkDetector)
# Facial Expression analysis library
add_subdirectory(lib/local/FaceAnalyser)
# Gaze estimation library
add_subdirectory(lib/local/GazeAnalyser)
# Utilities library
add_subdirectory(lib/local/Utilities)
# executables
add_subdirectory(exe/FaceLandmarkImg)
add_subdirectory(exe/FaceLandmarkVid)
add_subdirectory(exe/FaceLandmarkVidMulti)
add_subdirectory(exe/FeatureExtraction)
