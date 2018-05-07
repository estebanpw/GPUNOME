
// Standard utilities and common systems includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include "structs.h"

#define BUFFER_SIZE 2048
#define MAX_KERNEL_SIZE BUFFER_SIZE*100
#define CORES_PER_COMPUTE_UNIT 32

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void init_args(int argc, char ** av, FILE ** query, cl_uint * selected_device, ulong * z_value, ulong * kmer_size, FILE ** ref, FILE ** out, ulong * kmers_per_work_item, unsigned char * overlapping);
void print_hash_table(Hash_item * h);
char * get_basename(char * path);

int main(int argc, char ** argv)
{
    cl_uint selected_device = 0;
    ulong z_value = 1, kmer_size = 32;
    ulong kmers_per_work_item = 32;
    unsigned char overlapping = 1;
    FILE * query = NULL, * ref = NULL, * out = NULL;
    init_args(argc, argv, &query, &selected_device, &z_value, &kmer_size, &ref, &out, &kmers_per_work_item, &overlapping);

    ////////////////////////////////////////////////////////////////////////////////
    // Get info of devices
    ////////////////////////////////////////////////////////////////////////////////
    cl_platform_id platform_id = NULL;
    cl_device_id * devices = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_uint compute_units;
    char device_info[BUFFER_SIZE]; device_info[0] = '\0';
    cl_bool device_available;
    cl_ulong device_RAM;
    size_t work_group_size[3], work_group_size_global;
    cl_int ret;
    if(CL_SUCCESS != (ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms))){ fprintf(stderr, "Failed to get platforms\n"); exit(-1); }
    fprintf(stdout, "Detected %d platform(s)\n", ret_num_platforms);
    

    // Query how many devices there are
    if(CL_SUCCESS != (ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices))){ fprintf(stderr, "Failed to query number of devices\n"); exit(-1); }
    if ((devices = (cl_device_id *) malloc(sizeof(cl_device_id) * ret_num_devices)) == NULL){ fprintf(stderr, "Could not allocate devices information\n"); exit(-1); }

    // Query information about each device
    if(CL_SUCCESS != (ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, ret_num_devices, devices, &ret_num_devices))){ fprintf(stderr, "Failed to get devices information\n"); exit(-1); }
    
    fprintf(stdout, "Found %d device(s)\n", ret_num_devices);
    fprintf(stdout, "These are:\n");
    cl_uint i;
    for(i=0; i<ret_num_devices; i++){

        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, BUFFER_SIZE, device_info, NULL))) fprintf(stderr, "Failed to get device name %d\n", i);
        fprintf(stdout, "\tDevice [%d]: %s\n", i, device_info);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, BUFFER_SIZE, device_info, NULL))) fprintf(stderr, "Failed to get device profile %d\n", i);
        fprintf(stdout, "\t\tProfile      : %s\n", device_info);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_available, NULL))){ fprintf(stderr, "Failed to get device availability %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tis available?: %d\n", (int)device_available);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_RAM, NULL))){ fprintf(stderr, "Failed to get device global memory %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tGlobal mem   : %"PRIu64" (%"PRIu64" MB)\n", (uint64_t) device_RAM, (uint64_t) device_RAM / (1024*1024));
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_RAM, NULL))){ fprintf(stderr, "Failed to get device local memory %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tLocal mem    : %"PRIu64" (%"PRIu64" KB)\n", (uint64_t) device_RAM, (uint64_t) device_RAM / (1024));
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL))){ fprintf(stderr, "Failed to get device local memory %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tCompute units: %"PRIu64"\n", (uint64_t) compute_units);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size_global, NULL))){ fprintf(stderr, "Failed to get device global work items size %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tMax work group size: %zu\n", work_group_size_global);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), &work_group_size, NULL))){ fprintf(stderr, "Failed to get device work items size %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tWork size items: (%zu, %zu, %zu)\n", work_group_size[0], work_group_size[1], work_group_size[2]);
    }

    fprintf(stdout, "[INFO] Using device %d\n", selected_device);
    
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &devices[selected_device], NULL, NULL, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not create context. Error: %d\n", ret); exit(-1); }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, devices[selected_device], 0, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not create command queue. Error: %d\n", ret); exit(-1); }

    

    ////////////////////////////////////////////////////////////////////////////////
    // Make index dictionary
    ////////////////////////////////////////////////////////////////////////////////

    // Read sequence size
    fseek(query, 0, SEEK_END);
    ulong query_len_bytes = (ulong) ftell(query);
    rewind(query);


    // Allocate memory in host
    char * query_mem_host = (char *) malloc(query_len_bytes * sizeof(char));
    if(query_mem_host == NULL){ fprintf(stderr, "Could not allocate host memory for query sequence\n"); exit(-1); }

    // Load sequence into ram
    if(query_len_bytes != fread(query_mem_host, sizeof(char), query_len_bytes, query)){ fprintf(stderr, "Read incorrect amount of bytes from query\n"); exit(-1); }
    fclose(query);
    
    // Allocate hash table
    ulong hash_table_size = (ulong) pow(4.0, FIXED_K);
    cl_mem hash_table_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, hash_table_size * sizeof(Hash_item), NULL, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for hash table in device. Error: %d\n", ret); exit(-1); }
    
    // Initialize hash table
    Hash_item empty_hash_item;
    memset(&empty_hash_item, 0x0, sizeof(Hash_item));
    empty_hash_item.pos_in_y = 0xFFFFFFFFFFFFFFFF;
    ret = clEnqueueFillBuffer(command_queue, hash_table_mem, (const void *) &empty_hash_item, sizeof(Hash_item), 0, hash_table_size * sizeof(Hash_item), 0, NULL, NULL); 
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not initialize hash table. Error: %d\n", ret); exit(-1); }

    
    // Allocate memory in device
    cl_mem query_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query_len_bytes * sizeof(char), query_mem_host, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for query sequence in device. Error: %d\n", ret); exit(-1); }

    // Load kernel
    FILE * read_kernel; 
    switch(overlapping){
        case 0: { read_kernel = fopen("kernel_index_no_overlap.cl", "r"); fprintf(stdout, "[INFO] Using non overlapping k-mers\n"); } 
        break;
        case 1: { read_kernel = fopen("kernel_index.cl", "r"); fprintf(stdout, "[INFO] Using overlapping k-mers\n"); }
        break;
        default: { fprintf(stderr, "Bad choice of overlapping\n"); exit(-1); }
        break;
    }
    if(!read_kernel){ fprintf(stderr, "Failed to load kernel (1).\n"); exit(-1); }
    char * source_str = (char *) malloc(MAX_KERNEL_SIZE);
    if(source_str == NULL) { fprintf(stderr, "Could not allocate kernel\n"); exit(-1); }
    size_t source_size = fread(source_str, 1, MAX_KERNEL_SIZE, read_kernel);
    fclose(read_kernel);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating program (1): %d\n", ret); exit(-1); }

    // Build the program
    ret = clBuildProgram(program, 1, &devices[selected_device], NULL, NULL, NULL);
    if(ret != CL_SUCCESS){ 
        fprintf(stderr, "Error building program (1): %d\n", ret); 
        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *) malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            fprintf(stdout, "%s\n", log);
        }
        exit(-1); 
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "kernel_index", &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating kernel (1): %d\n", ret); exit(-1); }

    // Set working sizes
    fprintf(stdout, "[INFO] Query len size: %"PRIu64"\n", query_len_bytes);
    size_t local_item_size = CORES_PER_COMPUTE_UNIT * 8; // Number of work items in a work group
    size_t global_item_size;
    switch(overlapping){
        case 0: { 
            global_item_size = ((query_len_bytes - kmer_size + 1) / 32) / kmers_per_work_item;
        } 
        break;
        case 1: { 
            global_item_size = (query_len_bytes - kmer_size + 1) / kmers_per_work_item;
        }
        break;
    }
     
    global_item_size = global_item_size - (global_item_size % local_item_size); // Make it evenly divisable

    fprintf(stdout, "[INFO] Work items: %"PRIu64". Work groups: %"PRIu64". Total K-mers to be computed %"PRIu64"\n", (uint64_t) global_item_size, (uint64_t)(global_item_size/local_item_size), global_item_size * kmers_per_work_item);

    // Load parameters
    Parameters params = {z_value, kmer_size, query_len_bytes, (ulong) global_item_size, (ulong) kmers_per_work_item};    
    cl_mem params_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Parameters), &params, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for kmer sizes variable in device. Error: %d\n", ret); exit(-1); }


    // Set the arguments of the kernel
    //__kernel void kernel_index(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence)
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hash_table_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (1): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (2): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&query_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (3): %d\n", ret); exit(-1); }


    fprintf(stdout, "[INFO] Executing the kernel\n");
    // Execute the OpenCL kernel on the lists
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error enqueuing kernel (1): %d\n", ret); exit(-1); }


    // Wait for kernel to finish
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad flush of event: %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad finish of event: %d\n", ret); exit(-1); }
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }
    ret = clReleaseMemObject(query_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }
    ret = clReleaseMemObject(params_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (6)\n"); exit(-1); }
    free(query_mem_host);

    fprintf(stdout, "[INFO] Kernel execution finished. Code = %d\n", ret);


    // Hash_item * h = (Hash_item *) malloc(hash_table_size*sizeof(Hash_item));
    // if(h == NULL) { fprintf(stderr, "Could not allocate resulting hash table\n"); exit(-1); }
    // ret = clEnqueueReadBuffer(command_queue, hash_table_mem, CL_TRUE, 0, hash_table_size*sizeof(Hash_item), h, 0, NULL, NULL);
    // if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read from buffer: %d\n", ret); exit(-1); }
    // print_hash_table(h);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Match hits
    ////////////////////////////////////////////////////////////////////////////////


    
    // Read sequence size
    fseek(ref, 0, SEEK_END);
    ulong ref_len_bytes = (ulong) ftell(ref);
    rewind(ref);

    // Allocate memory in host
    char * ref_mem_host = (char *) malloc(ref_len_bytes * sizeof(char));
    if(ref_mem_host == NULL){ fprintf(stderr, "Could not allocate host memory for reference sequence\n"); exit(-1); }

    // Load sequence into ram
    if(ref_len_bytes != fread(ref_mem_host, sizeof(char), ref_len_bytes, ref)){ fprintf(stderr, "Read incorrect amount of bytes from query\n"); exit(-1); }
    fclose(ref);

    cl_mem ref_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ref_len_bytes * sizeof(char), ref_mem_host, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for reference sequence in device. Error: %d\n", ret); exit(-1); }
    
    
    // Load new kernel
    switch(z_value){
        case 1: read_kernel = fopen("kernel_match1.cl", "r");
        break;
        case 4: read_kernel = fopen("kernel_match2.cl", "r");
        break;
        case 8: read_kernel = fopen("kernel_match3.cl", "r");
        break;
        default: { fprintf(stderr, "Could not find kernel for z=%lu.\n", z_value); exit(-1); }
        break;
    }
    
    if(!read_kernel){ fprintf(stderr, "Failed to load kernel (2).\n"); exit(-1); }
    source_str[0] = '\0';
    source_size = fread(source_str, 1, MAX_KERNEL_SIZE, read_kernel);
    fclose(read_kernel);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating program (2): %d\n", ret); exit(-1); }
    
    // Build the program
    ret = clBuildProgram(program, 1, &devices[selected_device], NULL, NULL, NULL);
    if(ret != CL_SUCCESS){ 
        fprintf(stderr, "Error building program (2): %d\n", ret); 
        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *) malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(program, devices[selected_device], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            fprintf(stdout, "%s\n", log);
        }
        exit(-1); 
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "kernel_match", &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating kernel (2): %d\n", ret); exit(-1); }

    // Set working sizes
    global_item_size = (ref_len_bytes - kmer_size + 1) / kmers_per_work_item ; // Each work item corresponds to several kmers
    global_item_size = global_item_size - (global_item_size % local_item_size); // Make it evenly divisable (yes, this makes some kmers forgotten)

    fprintf(stdout, "[INFO] Work items: %"PRIu64". Work groups: %"PRIu64". Total K-mers to be computed %"PRIu64"\n", (uint64_t) global_item_size, (uint64_t)(global_item_size/local_item_size), global_item_size * kmers_per_work_item);

    // Set new parameters
    params.seq_length = ref_len_bytes;
    params.t_work_items = global_item_size;
    cl_mem params_mem_ref = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Parameters), &params, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for kmer sizes variable in device. Error: %d\n", ret); exit(-1); }

    // Set the arguments of the kernel
    //__kernel void kernel_index(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence)
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hash_table_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (1): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_mem_ref);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (2): %d\n", ret); exit(-1); }
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&ref_mem);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Bad setting of param (3): %d\n", ret); exit(-1); }

    fprintf(stdout, "[INFO] Executing the kernel\n");
    // Execute the OpenCL kernel on the lists
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error enqueuing kernel (2): %d\n", ret); exit(-1); }


    // Wait for kernel to finish
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad flush of event (2): %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad finish of event (2): %d\n", ret); exit(-1); }
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2.3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2.4)\n"); exit(-1); }
    ret = clReleaseMemObject(ref_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2.5)\n"); exit(-1); }
    free(ref_mem_host);


    ////////////////////////////////////////////////////////////////////////////////
    // Plot hits
    ////////////////////////////////////////////////////////////////////////////////


    fprintf(stdout, "[INFO] Kernel execution finished. Code = %d\n", ret);

    fprintf(stdout, "[INFO] Retrieving hash table. \n");
    Hash_item * h = (Hash_item *) malloc(hash_table_size*sizeof(Hash_item));
    if(h == NULL) { fprintf(stderr, "Could not allocate resulting hash table\n"); exit(-1); }
    ret = clEnqueueReadBuffer(command_queue, hash_table_mem, CL_TRUE, 0, hash_table_size*sizeof(Hash_item), h, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read from buffer: %d\n", ret); exit(-1); }

    // Scan hits table
    fprintf(stdout, "[INFO] Scanning hits table\n");

    //print_hash_table(h);

    ulong dimension = 1000;
    ulong idx;
    uint64_t ** representation = (uint64_t **) calloc(dimension+1, sizeof(uint64_t *));
    if(representation == NULL){ fprintf(stderr, "Could not allocate representation"); exit(-1); }
    for(idx=0; idx<dimension+1; idx++){
        representation[idx] = (uint64_t *) calloc(dimension+1, sizeof(uint64_t));
        if(representation[idx] == NULL){ fprintf(stderr, "Could not allocate second loop representation"); exit(-1); }
    }

    double ratio_query = (double) query_len_bytes / dimension;
    double ratio_ref = (double) ref_len_bytes / dimension;
    double pixel_size_query = (double) dimension / (double) query_len_bytes;
    double pixel_size_ref = (double) dimension / (double) ref_len_bytes;
    double i_r_fix = MAX(1.0, kmer_size * pixel_size_query);
    double j_r_fix = MAX(1.0, kmer_size * pixel_size_ref);
    for(idx=0; idx<hash_table_size; idx++){
        if(h[idx].repeat == 2 && h[idx].pos_in_y < 0xFFFFFFFFFFFFFFFF && h[idx].pos_in_x > 0){
            // Plot it  
            // Convert scale to representation
            //printf("With PX: %"PRIu64" and PY:%"PRIu64" (RX, RY) %e %e\n", h[idx].pos_in_x, h[idx].pos_in_y, ratio_ref, ratio_query);
            uint64_t redir_ref = (uint64_t) ((double)h[idx].pos_in_y / (ratio_ref));
            uint64_t redir_query = (uint64_t) ((double)h[idx].pos_in_x / (ratio_query));
            //printf("Writing at %"PRIu64", %"PRIu64"\n", redir_query, redir_ref);
            //getchar();
            double i_r = i_r_fix; double j_r = j_r_fix;
            while((uint64_t) i_r >= 1 && (uint64_t) j_r >= 1){
                if((int64_t) redir_query - (int64_t) i_r > 0 && (int64_t) redir_ref - (int64_t) j_r > 0){
                    representation[(int64_t) redir_query - (int64_t) i_r][(int64_t) redir_ref - (int64_t) j_r]++;
                }else{
                    if(redir_query > dimension || redir_ref > dimension) fprintf(stderr, "Exceeded dimension: %"PRIu64", %"PRIu64"\n", redir_query, redir_ref);
                    representation[redir_query][redir_ref]++;
                    break;
                }
                i_r -= MIN(1.0, pixel_size_query);
                j_r -= MIN(1.0, pixel_size_ref);
            }                                                     
        }
    }

    uint64_t unique_diffuse = 0;
    ulong j;
    for(i=0; i<dimension+1; i++){
        for(j=0; j<dimension; j++){
            fprintf(out, "%"PRIu64" ", representation[i][j]);
	        unique_diffuse += representation[i][j];
        }
        fprintf(out, "%"PRIu64"\n",  representation[i][dimension]);
	    unique_diffuse += representation[i][dimension];
    }

    fclose(out);

    free(h);
    
    for(j=0;j<dimension+1;j++){
        free(representation[j]);
    }
    free(representation);
    

    fprintf(stdout, "[INFO] Found %"PRIu64" unique hits for z = %"PRIu64".\n", unique_diffuse, z_value);


    // print_hash_table(h);
    
    // Close and deallocate everything
    //ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (1): %d\n", ret); exit(-1); }
    //ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2): %d\n", ret); exit(-1); }
    //ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    //ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }
    //ret = clReleaseMemObject(ref_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }
    ret = clReleaseMemObject(hash_table_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (6)\n"); exit(-1); }
    ret = clReleaseMemObject(params_mem_ref); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (7)\n"); exit(-1); }
    ret = clReleaseCommandQueue(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    ret = clReleaseContext(context); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    
    

    return 0;
}


void init_args(int argc, char ** av, FILE ** query, cl_uint * selected_device, ulong * z_value, ulong * kmer_size, FILE ** ref, FILE ** out, ulong * kmers_per_work_item, unsigned char * overlapping){
    
    int pNum = 0;
    char * p1, * p2;
    char outname[2048];
    while(pNum < argc){
        if(strcmp(av[pNum], "--help") == 0){
            fprintf(stdout, "USAGE:\n");
            fprintf(stdout, "           load_seq -query [file] -ref [file]\n");
            fprintf(stdout, "OPTIONAL:\n");
            
            fprintf(stdout, "           -dev        [Integer: d>=0] Selects the device to be used\n");
            fprintf(stdout, "           -kmer       [Integer: k>=1] Size of K-mer to be used\n");
            fprintf(stdout, "           -kwi        [Integer: k>=1] Number of kmers per work item to be used\n");
            fprintf(stdout, "           -diff       [Integer: z>=1] Inexactness applied\n");
            fprintf(stdout, "           --no-overlap   Turns overlap off in the indexing stage\n");
            fprintf(stdout, "           --help      Shows help for program usage\n");
            fprintf(stdout, "\n");
            exit(1);
        }

        if(strcmp(av[pNum], "--no-overlap") == 0){
            *overlapping = 0;
        }

        if(strcmp(av[pNum], "-query") == 0){
            *query = fopen(av[pNum+1], "rt");
            if(*query==NULL){ fprintf(stderr, "Could not open query file\n"); exit(-1); }
            p1 = get_basename(av[pNum+1]);
        }
        
        if(strcmp(av[pNum], "-ref") == 0){
            *ref = fopen(av[pNum+1], "rt");
            if(*ref==NULL){ fprintf(stderr, "Could not open reference file\n"); exit(-1); }
            p2 = get_basename(av[pNum+1]);
        }

        if(strcmp(av[pNum], "-dev") == 0){
            *selected_device = (cl_uint) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) < 0) { fprintf(stderr, "Device must be >0\n"); exit(-1); }
        }

        if(strcmp(av[pNum], "-kwi") == 0){
            *kmers_per_work_item = (ulong) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) < 1) { fprintf(stderr, "Kmers per work item size must be >0\n"); exit(-1); }
        }

        if(strcmp(av[pNum], "-kmer") == 0){
            *kmer_size = (ulong) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) <= 0) { fprintf(stderr, "Kmer size must be >0\n"); exit(-1); }
        }

        if(strcmp(av[pNum], "-diff") == 0){
            *z_value = (ulong) atoi(av[pNum+1]);
            if(*z_value != 1 && *z_value != 4 && *z_value != 8) { fprintf(stderr, "Z-value must be 1, 4 or 8\n"); exit(-1); }
        }

        pNum++;

    }   
    
    if(*query==NULL || *ref==NULL){ fprintf(stderr, "You have to include a query and a reference sequence!\n"); exit(-1); }
    strcat(outname, p1);
    strcat(outname, "-");
    strcat(outname, p2);
    strcat(outname, ".mat");
    *out = fopen(outname, "wt");
    if(*out == NULL){ fprintf(stderr, "Could not open output file\n"); exit(-1); }
    free(p1);
    free(p2);   
}


void print_hash_table(Hash_item * h){
    ulong i, sum = 0;
    for(i=0; i<pow(4, FIXED_K); i++){
        sum += h[i].repeat;
    }
    fprintf(stdout, "Sum is %lu\n", sum);
    for(i=0; i<pow(4, FIXED_K); i++){
        if(1 || (h[i].repeat == 2 && h[i].pos_in_x > 0 && h[i].pos_in_y < 0xFFFFFFFFFFFFFFFF )){
            /*
            fprintf(stdout, "#%lu: [b]%u%u%u%u%u%u%u%u [R]%lu [K]%lu [PX]%lu [PY]%lu\n", i, h[i].bitmask[0], 
            h[i].bitmask[1], h[i].bitmask[2], h[i].bitmask[3], h[i].bitmask[4], h[i].bitmask[5], 
            h[i].bitmask[6], h[i].bitmask[7], h[i].repeat, h[i].key, h[i].pos_in_x, h[i].pos_in_y);
            */
            fprintf(stdout, "#%lu: [R]%lu [K]%lu [PX]%lu [PY]%lu\n", i, h[i].repeat, h[i].key, h[i].pos_in_x, h[i].pos_in_y);
            getchar();
        }
        
        //if(h[i].key != 0) fprintf(stdout, "#%lu: [R]%lu [K]%lu [P]%lu\n", i, h[i].repeat, h[i].key, h[i].pos);
    }
}

char * get_basename(char * path){
    char * s = strrchr(path, '/');
    if (!s) return strdup(path); else return strdup(s + 1);
}