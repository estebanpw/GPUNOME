
CC=gcc
CXX=g++
OPENCL=/usr/local/cuda-8.0/include/
CFLAGS=-O3 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE -Wall #-DVERBOSE
BIN=.

all: index_kmers ligh_index_kmers



index_kmers: index_kmers.c
	$(CXX) $(CFLAGS) -I$(OPENCL) index_kmers.c -l OpenCL -o $(BIN)/index_kmers

ligh_index_kmers: index_kmers_split.c
	$(CXX) $(CFLAGS) -I$(OPENCL) index_kmers_split.c -l OpenCL -o $(BIN)/light_index_kmers

