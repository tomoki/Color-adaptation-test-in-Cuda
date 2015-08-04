#include "kernel.h"
#include <stdio.h>

int divRoundUp(int a, int b){
    return (a+b-1) / b;
}

void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}

__global__
void rgb_invert_all(uchar4* dst, size_t dst_pitch,
                    uchar4* src, size_t src_pitch,
                    int width, int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if((gidx < width) && (gidy < height)){
        size_t src_pos    = gidx+gidy*src_pitch/sizeof(uchar4);
        size_t dst_pos    = gidx+gidy*dst_pitch/sizeof(uchar4);

        uchar4 src_pixel = src[src_pos];
        unsigned char max_rgb = max(src_pixel.x,max(src_pixel.y,src_pixel.z));
        unsigned char min_rgb = min(src_pixel.x,min(src_pixel.y,src_pixel.z));

        dst[dst_pos].x = max_rgb - src_pixel.x + min_rgb;
        dst[dst_pos].y = max_rgb - src_pixel.y + min_rgb;
        dst[dst_pos].z = max_rgb - src_pixel.z + min_rgb;
        dst[dst_pos].w = src_pixel.w;
    }
}


// http://on-demand.gputechconf.com/gtc/2014/jp/sessions/4002.pdf
void rgb_invert_in_cuda(unsigned char* dst, unsigned char* src, int width, int height){
    uchar4* src_uchar;
    uchar4* dst_uchar;
    size_t src_pitch, dst_pitch;
    dim3 blockDim(64,2);
    dim3 gridDim(divRoundUp(width, blockDim.x), divRoundUp(height, blockDim.y));

    uchar4* uchar_buffer_in_cpu = (uchar4*)malloc(width*height*sizeof(uchar4));
    for(int i=0;i<width*height;i++){
        uchar_buffer_in_cpu[i].x = src[4*i+0];
        uchar_buffer_in_cpu[i].y = src[4*i+1];
        uchar_buffer_in_cpu[i].z = src[4*i+2];
        uchar_buffer_in_cpu[i].w = src[4*i+3];
    }

    getError(cudaMallocPitch(&src_uchar, &src_pitch, width*sizeof(uchar4), height));
    getError(cudaMallocPitch(&dst_uchar, &dst_pitch, width*sizeof(uchar4), height));
    getError(cudaMemcpy2D(src_uchar, src_pitch,
                          uchar_buffer_in_cpu, sizeof(uchar4)*width,
                          sizeof(uchar4)*width, height,
                          cudaMemcpyHostToDevice));
    rgb_invert_all<<<gridDim, blockDim>>>(dst_uchar, dst_pitch,
                                          src_uchar, src_pitch,
                                          width, height);
    getError(cudaMemcpy2D(uchar_buffer_in_cpu, sizeof(uchar4)*width,
                          dst_uchar, dst_pitch,
                          sizeof(uchar4)*width, height,
                          cudaMemcpyDeviceToHost));
    getError(cudaDeviceSynchronize());

    for(int i=0;i<width*height;i++){
        dst[4*i+0] = uchar_buffer_in_cpu[i].x;
        dst[4*i+1] = uchar_buffer_in_cpu[i].y;
        dst[4*i+2] = uchar_buffer_in_cpu[i].z;
        dst[4*i+3] = uchar_buffer_in_cpu[i].w;
    }
    free(uchar_buffer_in_cpu);
}