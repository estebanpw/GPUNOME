#include <CL/cl.h>

#define FIXED_K 12

typedef struct parameters{
    ulong z_value;
    ulong kmer_size;
	ulong seq_length;
	ulong n_cores;
} Parameters;

typedef struct hash_item{
    unsigned char bitmask[8];
    ulong repeat;
    ulong key;
    ulong pos;
} Hash_item;