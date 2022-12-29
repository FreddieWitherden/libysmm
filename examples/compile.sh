#!/bin/bash

gcc smm.c -o smm -I../include -L../build -lysmm-cl -lOpenCL
