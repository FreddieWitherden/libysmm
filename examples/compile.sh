#!/bin/bash

gcc smm.c -I../include -L../build -lysmm-cl -lOpenCL
cd ..
cd build
make
cd ..
cd examples
