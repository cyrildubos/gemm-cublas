#!/bin/bash
gcc -o gemm_cpu src/main.c -l cblas -l blas