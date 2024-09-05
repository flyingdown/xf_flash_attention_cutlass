# Install script for directory: /root/flash-attention/paged_attention/xf_flash_attention_cutlass

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/opt/dtk-24.04.1/llvm/bin/llvm-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib" TYPE SHARED_LIBRARY FILES "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/libpaged-attention.so")
  if(EXISTS "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so"
         OLD_RPATH "/lib/intel64:/lib/intel64_win:/lib/win-x64:/usr/local/lib/python3.10/site-packages/torch/lib:/usr/local/lib:/opt/dtk/roctracer/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/dtk-24.04.1/llvm/bin/llvm-strip" "$ENV{DESTDIR}/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/lib/libpaged-attention.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/include/paged_attn.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/pa_install/include" TYPE FILE FILES "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/csrc/paged_attn.h")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/root/flash-attention/paged_attention/xf_flash_attention_cutlass/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
