# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra ")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files (x86)/onnxruntime")
  else()
    include_directories("/home/eekmekci/Downloads/onnxruntime-inference-examples-cxx-for-linux-main/c_cxx/include")
  endif()
endif()

include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

#if JPEG lib is available, we'll use it for image decoding, otherwise we'll use WIC
find_package(JPEG)
if(LIBPNG_ROOTDIR)
  set(PNG_FOUND true)
  if(WIN32)
    set(PNG_LIBRARIES debug libpng16_d optimized libpng16)
  else()
    set(PNG_LIBRARIES png16)
  endif()
  set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
  set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
else()
  find_package(PNG)
endif()

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
  add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

# some examples require a Windows build environment
if(PNG_FOUND)
  add_subdirectory(onnx_inference_color)
  
endif()
#missing experimental_onnxruntime_cxx_api.h
#add_subdirectory(model-explorer)
