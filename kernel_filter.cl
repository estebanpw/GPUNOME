// #define DIMENSION 1000
#define DIAG_LEN 4
#define DIAG_EXTEND 5
#define DIST_TH 1.5
#define NOISE 1
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Reduce this to 1D
__kernel void kernel_filter(__global const int * m_in, __global int * m_out, __global ulong * dimension) {
 
    // Get the index of the current element to be processed
	int x_id = (int) get_global_id(0);
	int y_id = (int) get_global_id(1);

	ulong sum = 0;
	ulong DIMENSION = *dimension;
	
	// Kernel to improve diagonals (forward)

	if(x_id > DIAG_EXTEND && x_id < DIMENSION-DIAG_EXTEND && y_id > DIAG_EXTEND && y_id < DIMENSION-DIAG_EXTEND){
		
		sum += (ulong) m_in[(x_id - 2)*DIMENSION + y_id - 2];
		sum += (ulong) m_in[(x_id - 1)*DIMENSION + y_id - 1];
		sum += (ulong) m_in[(x_id)*DIMENSION + y_id];
		sum += (ulong) m_in[(x_id + 1)*DIMENSION + y_id + 1];
		sum += (ulong) m_in[(x_id + 2)*DIMENSION + y_id + 2];


		if(sum >= DIAG_LEN){
			ulong k;
			for(k=0; k<DIAG_EXTEND; k++){
				//m_out[(x_id+k)*DIMENSION + y_id+k] = 1;
				atomic_inc(&m_out[(x_id+k)*DIMENSION + y_id+k]);
				//m_out[(x_id-k)*DIMENSION + y_id-k] = 1;
				atomic_inc(&m_out[(x_id-k)*DIMENSION + y_id-k]);
			}
		}else{
			//m_out[(x_id)*DIMENSION + y_id] = 0;
			//atomic_dec(&m_out[(x_id)*DIMENSION + y_id]);
		}
	}

	// Kernel to improve diagonals (reverse)

	sum = 0;

	if(x_id > DIAG_EXTEND && x_id < DIMENSION-DIAG_EXTEND && y_id > DIAG_EXTEND && y_id < DIMENSION-DIAG_EXTEND){
		
		sum += (ulong) m_in[(x_id + 2)*DIMENSION + y_id - 2];
		sum += (ulong) m_in[(x_id + 1)*DIMENSION + y_id - 1];
		sum += (ulong) m_in[(x_id)*DIMENSION + y_id];
		sum += (ulong) m_in[(x_id - 1)*DIMENSION + y_id + 1];
		sum += (ulong) m_in[(x_id - 2)*DIMENSION + y_id + 2];


		if(sum >= DIAG_LEN){
			ulong k;
			for(k=0; k<DIAG_EXTEND; k++){
				//m_out[(x_id-k)*DIMENSION + y_id+k] = 1;
				atomic_inc(&m_out[(x_id-k)*DIMENSION + y_id+k]);
				//m_out[(x_id+k)*DIMENSION + y_id-k] = 1;
				atomic_inc(&m_out[(x_id+k)*DIMENSION + y_id-k]);
			}
		}else{
			//m_out[(x_id)*DIMENSION + y_id] = 0;
			//atomic_dec(&m_out[(x_id)*DIMENSION + y_id]); 
		}
	}


	// Kernel to remove noise
	
	sum = 0;

	// Concurrent-problems free
	if(x_id >= 1 && x_id <= DIMENSION-1 && y_id >= 1 && y_id <= DIMENSION-1){
		
		int min_i = MAX(1, x_id - 1);
		int max_i = MIN(DIMENSION-1, x_id + 1);
		int min_y = MAX(1, y_id - 1);
		int max_y = MIN(DIMENSION-1, y_id + 1);

		sum += (ulong) m_in[(min_i)*DIMENSION + min_y];
		sum += (ulong) m_in[(min_i)*DIMENSION + y_id];
		sum += (ulong) m_in[(min_i)*DIMENSION + max_y];

		sum += (ulong) m_in[(x_id)*DIMENSION + min_y];
		sum += (ulong) m_in[(x_id)*DIMENSION + y_id];
		sum += (ulong) m_in[(x_id)*DIMENSION + max_y];

		sum += (ulong) m_in[(max_i)*DIMENSION + min_y];
		sum += (ulong) m_in[(max_i)*DIMENSION + y_id];
		sum += (ulong) m_in[(max_i)*DIMENSION + max_y];

		//if(sum > NOISE) m_out[(x_id)*DIMENSION + y_id] = 1;
		if(sum > NOISE) atomic_inc(&m_out[(x_id)*DIMENSION + y_id]);
	}

	

}
