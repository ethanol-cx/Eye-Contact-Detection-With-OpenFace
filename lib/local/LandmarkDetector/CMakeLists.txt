#TBB library
include_directories(${TBB_ROOT_DIR}/include)

include_directories(${BOOST_INCLUDE_DIR})

set(CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_CONFIG_DIR}")
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/include/PathConfig.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/PathConfig.h
	NEWLINE_STYLE WIN32
)

SET(SOURCE
    src/CCNF_patch_expert.cpp
	src/LandmarkDetectionValidator.cpp
    src/LandmarkDetectorFunc.cpp
	src/LandmarkDetectorModel.cpp
    src/LandmarkDetectorUtils.cpp
	src/LandmarkDetectorParameters.cpp
	src/Patch_experts.cpp
	src/PAW.cpp
    src/PDM.cpp
	src/SVR_patch_expert.cpp
	src/stdafx.cpp
)

SET(HEADERS
    include/CCNF_patch_expert.h	
    include/LandmarkCoreIncludes.h
	include/LandmarkDetectionValidator.h
    include/LandmarkDetectorFunc.h
	include/LandmarkDetectorModel.h
	include/LandmarkDetectorParameters.h
	include/LandmarkDetectorUtils.h
	include/Patch_experts.h	
    include/PAW.h
	include/PDM.h
	include/SVR_patch_expert.h		
	include/stdafx.h
)

SET(EXTRA_HEADERS
	include/PathConfig.h
)

include_directories(./include)
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

add_library( LandmarkDetector ${SOURCE} ${HEADERS} ${EXTRA_HEADERS})

install (TARGETS LandmarkDetector DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
