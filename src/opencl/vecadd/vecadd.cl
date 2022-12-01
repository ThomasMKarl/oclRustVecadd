#ifndef ARRAY_TYPE
//#warning "type of vector elements not specified, defaulting to float"
#define ARRAY_TYPE float
#endif

#pragma OPENCL EXTENSION cl_khr_fp16: enable

__kernel void addVectors(__global const ARRAY_TYPE *a, __global const ARRAY_TYPE *b, __global ARRAY_TYPE *c, ulong num) {
  unsigned long long int gid = get_global_id(0);
  if(gid < num)
    c[gid] = a[gid] + b[gid];
}

__kernel void addVectorsInplace(__global ARRAY_TYPE *a, __global const ARRAY_TYPE *b, ulong num) {
  unsigned long long int gid = get_global_id(0);
  if(gid < num)
    a[gid] += b[gid];
}

__kernel void subVectors(__global const ARRAY_TYPE *a, __global const ARRAY_TYPE *b, __global ARRAY_TYPE *c, ulong num) {
  unsigned long long int gid = get_global_id(0);
  if(gid < num)
    c[gid] = a[gid] - b[gid];
}

__kernel void subVectorsInplace(__global ARRAY_TYPE *a, __global const ARRAY_TYPE *b, ulong num) {
  unsigned long long int gid = get_global_id(0);
  if(gid < num)
    a[gid] -= b[gid];
}