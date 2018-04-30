
CC=gcc
CXX=g++
OPENCL=/usr/local/cuda-8.0/include/
CFLAGS=-g -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -Wall #-DVERBOSE
BIN=.

all: index_kmers



index_kmers: index_kmers.c
	$(CXX) $(CFLAGS) -I$(OPENCL) index_kmers.c -l OpenCL -o $(BIN)/index_kmers



