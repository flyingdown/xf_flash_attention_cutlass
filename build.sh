#!/bin/bash

#make clean_all
#cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DAMDGPU_TARGETS=gfx928 .. 
#make -j8 VERBOSE=1

hipcc -o test test.cc -lpaged-attention -Lbuild -Icsrc/ -Icsrc/flash_attn -Icsrc/flash_attn/src -Icsrc/cutlass_3.2.1/include/
