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

__constant ulong pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L, 
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L, 
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L, 
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L, 
    72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};

__kernel void kernel_index(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence) {
 
    // Get the index of the current element to be processed
    ulong i = get_global_id(0);

	
	ulong local_k_size = params->kmer_size, j, k;
	
	// Until reaching end of sequence
	for(j=0; j<(params->seq_length/params->n_cores); j++){
		// Calculate next position so that access is coalescent
		ulong pos = i + params->n_cores * j;
		ulong hash12 = 0, hash_full = 0;
		
		for(k=0; k<FIXED_K; k++){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase
			switch(sequence[pos+k]){
				case 'A': {}
				break;
				case 'C': hash12 += pow4[k]; 
				break;
				case 'G': hash12 += pow4[k] * 2; 
				break;
				case 'T': hash12 += pow4[k] * 3;
				break;
				default: {}
				break;
			}
		}

		hash_full = hash12;
		
		for(k=FIXED_K; k<32; k++){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase
			switch(sequence[pos+k]){
				case 'A': {}
				break;
				case 'C': hash_full += pow4[k]; 
				break;
				case 'G': hash_full += pow4[k] * 2; 
				break;
				case 'T': hash_full += pow4[k] * 3;
				break;
				default: {}
				break;
			}
		}


		
		// Index with prefix
		hash_table[hash12].key = hash_full;
		hash_table[hash12].pos = pos;
		++(hash_table[hash12].repeat);
		hash_table[hash12].bitmask[i % 8] = (unsigned char) 1;
	}

}
