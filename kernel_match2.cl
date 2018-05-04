// This kernel uses z = 4

#define FIXED_K 12
#define ZVAL 4

typedef struct parameters{
    ulong z_value;
    ulong kmer_size;
	ulong seq_length;
	ulong t_work_items;
	ulong kmers_per_work_item;
} Parameters;

typedef struct hash_item{
    //unsigned char bitmask[8];
    ulong repeat;
    ulong key;
    ulong pos_in_x;
    ulong pos_in_y;
} Hash_item;

__constant ulong pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L, 
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L, 
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L, 
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L, 
    72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};

__kernel void kernel_match(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence) {
 
    // Get the index of the current element to be processed
	ulong global_id = get_global_id(0);
	//ulong local_id = get_local_id(0);
	ulong kmers_in_work_item = params->kmers_per_work_item;
	ulong t_work_items = params->t_work_items;
	ulong kmer_size = params->kmer_size;
	ulong j, k;

	// Debugging
	//hash_table[0].repeat = i;
	//hash_table[1].repeat = local_size;
	
	// Until reaching end of sequence
	for(j=0; j<kmers_in_work_item; j++){
		
		// Coalescent
		ulong pos = global_id + (j * t_work_items);

		// Non coalescent (Naive approach)
		//ulong pos = (global_id * kmers_in_work_item) + j; 

		// Completely not coalescent
		//uint seed = global_id;
		//uint t = seed ^ (seed << 11);  
		//ulong pos = (ulong) ((local_id ^ (local_id >> 19) ^ (t ^ (t >> 8))) % params->seq_length);
		
		

		ulong hash12 = 0, hash12_rev = 0, hash_full = 0, hash_full_rev = 0;
		
		// Forward part for first 12
		unsigned char bad = 0;
		for(k=0; k<FIXED_K; k++){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase
			switch(sequence[pos+k]){
				case 'A': {  }
				break;
				case 'C': { hash12 += pow4[k]; }
				break;
				case 'G': { hash12 += pow4[k] * 2; }
				break;
				case 'T': { hash12 += pow4[k] * 3; }
				break;
				case '\n': {  }
				break;
				default: { bad = 1; }
				break;
			}
		}
		
		hash_full = hash12;

		// Forward part for non indexing
		for(k=FIXED_K; k<kmer_size; k+=ZVAL){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase
			switch(sequence[pos+k]){
				case 'A': {   }
				break;
				case 'C': { hash_full += pow4[k];  }
				break;
				case 'G': { hash_full += pow4[k] * 2;  }
				break;
				case 'T': { hash_full += pow4[k] * 3; }
				break;
				case '\n': {  }
				break;
				default: { bad = 1; }
				break;
			}
		}

		// Reverse part for non indexing nucleotides
		for(k=3; k<20; k+=ZVAL){
			switch(sequence[pos+k]){
				case 'A': { hash_full_rev += pow4[kmer_size - k - 1] * 3; }
				break;
				case 'C': { hash_full_rev += pow4[kmer_size - k - 1] * 2; }
				break;
				case 'G': { hash_full_rev += pow4[kmer_size - k - 1]; }
				break;
				case 'T': {  }
				break;
				case '\n': {  }
				break;
				default: { bad = 1; }
				break;
			}
		}

		// Reverse part for indexing nucleotides
		for(k=20; k<kmer_size; k++){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase
			switch(sequence[pos+k]){
				case 'A': { hash12_rev += pow4[kmer_size - k - 1] * 3; }
				break;
				case 'C': { hash12_rev += pow4[kmer_size - k - 1] * 2; }
				break;
				case 'G': { hash12_rev += pow4[kmer_size - k - 1]; }
				break;
				case 'T': {  }
				break;
				case '\n': {  }
				break;
				default: { bad = 1; }
				break;
			}
		}

		hash_full_rev += hash12_rev;


		if(bad == 0){
			// Index with prefix
			if(hash_table[hash12].key == hash_full){
				hash_table[hash12].pos_in_y = pos;
				++(hash_table[hash12].repeat);
			}

			// And reverse
			if(hash_table[hash12_rev].key == hash_full_rev){
				hash_table[hash12_rev].pos_in_y = pos;
				++(hash_table[hash12_rev].repeat);
			}
		}

		
		
		//hash_table[hash12].bitmask[pos % 8] = (unsigned char) 1;
		
	}

}
