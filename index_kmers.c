
// Standard utilities and common systems includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include "structs.h"

#define BUFFER_SIZE 2048
#define MAX_KERNEL_SIZE BUFFER_SIZE*100

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void init_args(int argc, char ** av, FILE ** query, cl_uint * selected_device, ulong * z_value, ulong * kmer_size);
void print_hash_table(Hash_item * h);

int main(int argc, char ** argv)
{
    cl_uint selected_device = 0;
    ulong z_value = 1, kmer_size = 32;
    FILE * query = NULL;
    init_args(argc, argv, &query, &selected_device, &z_value, &kmer_size);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id * devices = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
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
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &device_available, NULL))){ fprintf(stderr, "Failed to get device availability %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tis available?: %d\n", (int)device_available);
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_RAM, NULL))){ fprintf(stderr, "Failed to get device global memory %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tGlobal mem   : %"PRIu64" (%"PRIu64" MB)\n", (uint64_t) device_RAM, (uint64_t) device_RAM / (1024*1024));
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &device_RAM, NULL))){ fprintf(stderr, "Failed to get device local memory %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tLocal mem    : %"PRIu64" (%"PRIu64" KB)\n", (uint64_t) device_RAM, (uint64_t) device_RAM / (1024));
        if(CL_SUCCESS != (ret = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size_global, NULL))){ fprintf(stderr, "Failed to get device global work items size %d\n", i); exit(-1); }
        fprintf(stdout, "\t\tWork size global: %zu\n", work_group_size_global);
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
    // Program start
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
    
    // Allocate hash table
    ulong hash_table_size = (ulong) pow(4.0, FIXED_K);
    cl_mem hash_table_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, hash_table_size * sizeof(Hash_item), NULL, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for hash table in device. Error: %d\n", ret); exit(-1); }
    
    // Initialize hash table
    Hash_item empty_hash_item;
    memset(&empty_hash_item, 0x0, sizeof(Hash_item));
    ret = clEnqueueFillBuffer(command_queue, hash_table_mem, (const void *) &empty_hash_item, sizeof(Hash_item), 0, hash_table_size * sizeof(Hash_item), 0, NULL, NULL); 
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not initialize hash table. Error: %d\n", ret); exit(-1); }
  
    Parameters params = {z_value, kmer_size, query_len_bytes, (ulong) work_group_size_global};    
    cl_mem params_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Parameters), &params, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for kmer sizes variable in device. Error: %d\n", ret); exit(-1); }
    
    // Allocate memory in device
    cl_mem query_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, query_len_bytes * sizeof(char), query_mem_host, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not allocate memory for query sequence in device. Error: %d\n", ret); exit(-1); }

    // Load kernel
    FILE * read_kernel = fopen("kernel_index.cl", "r");
    if(!read_kernel){ fprintf(stderr, "Failed to load kernel.\n"); exit(-1); }
    char * source_str = (char *) malloc(MAX_KERNEL_SIZE);
    if(source_str == NULL) { fprintf(stderr, "Could not allocate kernel\n"); exit(-1); }
    size_t source_size = fread(source_str, 1, MAX_KERNEL_SIZE, read_kernel);
    fclose(read_kernel);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const size_t *) &source_size, &ret);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating program: %d\n", ret); exit(-1); }

    // Build the program
    ret = clBuildProgram(program, 1, &devices[selected_device], NULL, NULL, NULL);
    if(ret != CL_SUCCESS){ 
        fprintf(stderr, "Error building program: %d\n", ret); 
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
    if(ret != CL_SUCCESS){ fprintf(stderr, "Error creating kernel: %d\n", ret); exit(-1); }

    // Set working sizes
    size_t global_item_size = work_group_size_global; // Process the entire lists
    size_t local_item_size = 128; // Divide work items into groups of 64

    fprintf(stdout, "[INFO] work groups@ %d*%d\n", (int) (work_group_size_global / local_item_size) , (int)local_item_size);


    // Set the arguments of the kernel
    //__kernel void kernel_index(__global Hash_item * hash_table, __global Parameters * params, __global const char * sequence)
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hash_table_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&params_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&query_mem);


    fprintf(stdout, "[INFO] Executing the kernel\n");
    // Execute the OpenCL kernel on the lists
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);


    // Wait for kernel to finish
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad flush of event: %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad finish of event: %d\n", ret); exit(-1); }
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }
    ret = clReleaseMemObject(query_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }

    fprintf(stdout, "[INFO] Kernel execution finished. Code = %d\n", ret);
    
    /*
    // To read the index dictionary
    Hash_item * h = (Hash_item *) malloc(hash_table_size*sizeof(Hash_item));
    if(h == NULL) { fprintf(stderr, "Could not allocate resulting hash table\n"); exit(-1); }
    ret = clEnqueueReadBuffer(command_queue, hash_table_mem, CL_TRUE, 0, hash_table_size*sizeof(Hash_item), h, 0, NULL, NULL);
    if(ret != CL_SUCCESS){ fprintf(stderr, "Could not read from buffer: %d\n", ret); exit(-1); }
    print_hash_table(h);
    */
    
    fprintf(stdout, "[INFO] Size of query %lu\n", query_len_bytes);

    
    
    // Close and deallocate everything
    ret = clFlush(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (1): %d\n", ret); exit(-1); }
    ret = clFinish(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (2): %d\n", ret); exit(-1); }
    ret = clReleaseKernel(kernel); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (3)\n"); exit(-1); }
    ret = clReleaseProgram(program); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (4)\n"); exit(-1); }
    ret = clReleaseMemObject(query_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (5)\n"); exit(-1); }
    ret = clReleaseMemObject(hash_table_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (6)\n"); exit(-1); }
    ret = clReleaseMemObject(params_mem); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (7)\n"); exit(-1); }
    ret = clReleaseCommandQueue(command_queue); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    ret = clReleaseContext(context); if(ret != CL_SUCCESS){ fprintf(stderr, "Bad free (8)\n"); exit(-1); }
    
    
    free(query_mem_host);
    fclose(query);

    return 0;
}


void init_args(int argc, char ** av, FILE ** query, cl_uint * selected_device, ulong * z_value, ulong * kmer_size){
    
    int pNum = 0;
    while(pNum < argc){
        if(strcmp(av[pNum], "--help") == 0){
            fprintf(stdout, "USAGE:\n");
            fprintf(stdout, "           load_seq -query [file]\n");
            fprintf(stdout, "OPTIONAL:\n");
            
            fprintf(stdout, "           -dev        [Integer: d>=0] Selects the device to be used\n");
            fprintf(stdout, "           -kmer       [Integer: k>=1] Size of K-mer to be used\n");
            fprintf(stdout, "           -diff       [Integer: z>=1] Inexactness applied\n");
            fprintf(stdout, "           --help      Shows help for program usage\n");
            fprintf(stdout, "\n");
            exit(1);
        }
        if(strcmp(av[pNum], "-query") == 0){
            *query = fopen(av[pNum+1], "rt");
            if(*query==NULL){ fprintf(stderr, "Could not open query file\n"); exit(-1); }
        }
        
        if(strcmp(av[pNum], "-dev") == 0){
            *selected_device = (cl_uint) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) < 0) { fprintf(stderr, "Device must be >0\n"); exit(-1); }
        }

        if(strcmp(av[pNum], "-kmer") == 0){
            *kmer_size = (ulong) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) <= 0) { fprintf(stderr, "Kmer size must be >0\n"); exit(-1); }
        }

        if(strcmp(av[pNum], "-diff") == 0){
            *z_value = (ulong) atoi(av[pNum+1]);
            if(atoi(av[pNum+1]) < 1) { fprintf(stderr, "Z-value must be >0\n"); exit(-1); }
        }

        pNum++;

    }   
    
    if(*query==NULL){ fprintf(stderr, "You have to include a query sequence!\n"); exit(-1); }
}


void print_hash_table(Hash_item * h){
    ulong i;
    for(i=0; i<pow(4, FIXED_K); i++){
        if(h[i].repeat == 1){
            fprintf(stdout, "#%lu: [b]%u%u%u%u%u%u%u%u [R]%lu [K]%lu [P]%lu\n", i, h[i].bitmask[0], 
            h[i].bitmask[1], h[i].bitmask[2], h[i].bitmask[3], h[i].bitmask[4], h[i].bitmask[5], 
            h[i].bitmask[6], h[i].bitmask[7], h[i].repeat, h[i].key, h[i].pos);
            getchar();
        }
        
        //if(h[i].key != 0) fprintf(stdout, "#%lu: [R]%lu [K]%lu [P]%lu\n", i, h[i].repeat, h[i].key, h[i].pos);
    }
}
