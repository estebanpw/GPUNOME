#define FIXED_K 12

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct parameters{
    ulong z_value;
    ulong kmer_size;
	ulong seq_length;
	ulong t_work_items;
	ulong kmers_per_work_item;
	ulong offset;
} Parameters;

typedef struct hash_item{
    //unsigned char bitmask[8];
    ulong repeat;
    ulong key;
    ulong pos_in_x;
    ulong pos_in_y;
} Hash_item;

/*
__constant ulong pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L, 
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L, 
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L, 
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L, 
    72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};
*/

__kernel void kernel_index(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence) {
 
    // Get the index of the current element to be processed
	ulong global_id = get_global_id(0);
	//ulong local_id = get_local_id(0);
	ulong kmers_in_work_item = params->kmers_per_work_item;
	ulong t_work_items = params->t_work_items;
	ulong offset = params->offset;
	ulong j, k;

	/*
	unsigned char converter[(unsigned) 'T' + 1];
	for(j=0;j<(unsigned) 'T'+1; j++) converter[j] = 0;
	converter['A'] = 0; converter['C'] = 1; converter['G'] = 2; converter['T'] = 3; converter['N'] = 4;
	*/

	// Will this work?
	/*
	ulong pow4[33]={1L, 4L, 16L, 64L, 256L, 1024L, 4096L, 16384L, 65536L, 
    262144L, 1048576L, 4194304L, 16777216L, 67108864L, 268435456L, 1073741824L, 4294967296L, 
    17179869184L, 68719476736L, 274877906944L, 1099511627776L, 4398046511104L, 17592186044416L, 
    70368744177664L, 281474976710656L, 1125899906842624L, 4503599627370496L, 18014398509481984L, 
    72057594037927936L, 288230376151711744L, 1152921504606846976L, 4611686018427387904L};
	*/

	// Debugging
	//hash_table[0].repeat = i;
	//hash_table[1].repeat = local_size;
	

	// Until reaching end of sequence
	for(j=0; j<kmers_in_work_item; j++){
		
		// Coalescent
		ulong pos = (global_id * 32) + (j * 32 * t_work_items);
		// Non coalescent
		//ulong pos = (global_id * kmers_in_work_item * 32) + (j * 32);
		
		// Completely not coalescent
		//uint seed = global_id;
		//uint t = seed ^ (seed << 11);  
		//ulong pos = (ulong) ((local_id ^ (local_id >> 19) ^ (t ^ (t >> 8))) % params->seq_length);
		

		ulong hash12 = 0, hash_full = 0;
		unsigned char checker = 0, multiplier = 0, val;
		
		//unsigned char bad = 0;
		for(k=0; k<FIXED_K; k++){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase

			val = (unsigned char) sequence[pos+k];
			multiplier = (val & (unsigned char) 6) >> 1;
			//hash12 += (((ulong) 1) << (2*k)) * (ulong) converter[sequence[pos+k]];
			hash12 += (((ulong) 1) << (2*k)) * (ulong) multiplier;
			checker = checker | (val & (unsigned char) 8);

			/*

			switch(sequence[pos+k]){
				case 'A': {}
				break;
				//case 'C': hash12 += pow4[k]; 
				case 'C': hash12 += (((ulong) 1) << (2*k)); 
				break;
				case 'G': hash12 += (((ulong) 1) << (2*k)) * 2; 
				break;
				case 'T': hash12 += (((ulong) 1) << (2*k)) * 3;
				break;
				case '\n': { }
				break;
				default: { bad = 1; }
				break;
			}

			*/
		}

		hash_full = hash12;
		
		
		for(k=FIXED_K; k<params->kmer_size; k+=params->z_value){
			// Restriction: Make sure input sequences have no ">" lines and all letters are uppercase

			val = (unsigned char) sequence[pos+k];
			multiplier = (val & (unsigned char) 6) >> 1;
			hash_full += (((ulong) 1) << (2*k)) * (ulong) multiplier;
			checker = checker | (val & (unsigned char) 8);

			/*
			switch(sequence[pos+k]){
				case 'A': {}
				break;
				case 'C': hash_full += (((ulong) 1) << (2*k)); 
				break;
				case 'G': hash_full += (((ulong) 1) << (2*k)) * 2; 
				break;
				case 'T': hash_full += (((ulong) 1) << (2*k)) * 3;
				break;
				case '\n': { }
				break;
				default: { bad = 1; }
				break;
			}
			*/
		}
		

		
		//if(bad == 0){
		if(checker == (unsigned char) 0){
			// Index with prefix
			hash_table[hash12].key = hash_full;
			hash_table[hash12].pos_in_x = pos + offset;
			//++(hash_table[hash12].repeat);
			atom_inc(&hash_table[hash12].repeat);


			//hash_table[hash12].bitmask[pos % 8] = (unsigned char) 1;
		}	
		
		
	}
	
}
