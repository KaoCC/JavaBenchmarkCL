


__kernel void test(__global int* a, __global int* b, __global int* c) {

    int gid = get_global_id(0);

    a[gid] += b[gid];

    c[gid] = a[gid];


}