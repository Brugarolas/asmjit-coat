cmake_minimum_required(VERSION 3.5)

project(perf-util)

include_directories(./)



if (NOT WIN32)
	add_library(perf-util STATIC
		ittnotify/disable_warnings.h
		ittnotify/ittnotify_config.h
		ittnotify/ittnotify_static.c
		ittnotify/ittnotify_static.h
		ittnotify/ittnotify_types.h
		ittnotify/ittnotify.h
		ittnotify/jitprofiling.c
		ittnotify/jitprofiling.h
		jit_utils/jit_utils.cpp
		jit_utils/jit_utils.hpp
	)
	target_link_libraries(perf-util ${CMAKE_DL_LIBS})
else()
	add_library(perf-util STATIC
		ittnotify/disable_warnings.h
		ittnotify/ittnotify_config.h
		jit_utils/jit_utils.cpp
		jit_utils/jit_utils.hpp
	)
endif()
