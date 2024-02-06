#!/bin/bash
nvcc -o gemm_gpu src/main.cu -l cublas