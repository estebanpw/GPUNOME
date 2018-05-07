#define DIMENSION 1000
#define DIAG_LEN 4
#define DIAG_EXTEND 5
#define DIST_TH 1.5
#define NOISE 1


// Reduce this to 1D
__kernel void kernel_filter(__global const unsigned char ** m_in, __global unsigned char ** m_out) {
 
    // Get the index of the current element to be processed
	int x_id = (int) get_global_id(0);
	int y_id = (int) get_global_id(1);

	ulong sum = 0;
	
	// Kernel to improve diagonals (forward)

	if(x_id >= DIAG_EXTEND && x_id < DIMENSION-DIAG_EXTEND && y_id >= DIAG_EXTEND && y_id < DIMENSION-DIAG_EXTEND){
		
		sum += (ulong) m_in[x_id - 2][y_id - 2];
		sum += (ulong) m_in[x_id - 1][y_id - 1];
		sum += (ulong) m_in[x_id][y_id];
		sum += (ulong) m_in[x_id + 1][y_id + 1];
		sum += (ulong) m_in[x_id + 2][y_id + 2];


		if(sum > DIAG_LEN){
			ulong k;
			for(k=0; k<DIAG_EXTEND; k++){
				m_out[x_id+k][y_id+k] = 1;
				m_out[x_id-k][y_id-k] = 1;
			}
		}else{
			m_out[x_id][y_id] = 0;
		}
	}

	// Kernel to improve diagonals (reverse)

	if(x_id >= DIAG_EXTEND && x_id < DIMENSION-DIAG_EXTEND && y_id >= DIAG_EXTEND && y_id < DIMENSION-DIAG_EXTEND){
		
		sum += (ulong) m_in[x_id + 2][y_id - 2];
		sum += (ulong) m_in[x_id + 1][y_id - 1];
		sum += (ulong) m_in[x_id][y_id];
		sum += (ulong) m_in[x_id - 1][y_id + 1];
		sum += (ulong) m_in[x_id - 2][y_id + 2];


		if(sum > DIAG_LEN){
			ulong k;
			for(k=0; k<DIAG_EXTEND; k++){
				m_out[x_id-k][y_id+k] = 1;
				m_out[x_id+k][y_id-k] = 1;
			}
		}else{
			m_out[x_id][y_id] = 0;
		}
	}


	// Kernel to remove noise
	
	if(x_id >= 1 && x_id <= DIMENSION-1 && y_id >= 1 && y_id <= DIMENSION-1){
		sum = 0;
		sum += (ulong) m_in[x_id - 1][y_id - 1];
		sum += (ulong) m_in[x_id - 1][y_id];
		sum += (ulong) m_in[x_id - 1][y_id + 1];

		sum += (ulong) m_in[x_id][y_id - 1];
		sum += (ulong) m_in[x_id][y_id];
		sum += (ulong) m_in[x_id][y_id + 1];

		sum += (ulong) m_in[x_id + 1][y_id - 1];
		sum += (ulong) m_in[x_id + 1][y_id];
		sum += (ulong) m_in[x_id + 1][y_id + 1];

		if(sum > NOISE) m_out[x_id][y_id] = 1;
	}
}
