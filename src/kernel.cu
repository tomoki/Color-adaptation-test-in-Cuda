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
void rgb_invert_all(uchar4* mem, size_t mem_pitch,
                    int width, int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if((gidx < width) && (gidy < height)){
        size_t mem_pos    = gidx+gidy*mem_pitch/sizeof(uchar4);
        uchar4 src_pixel = mem[mem_pos];

        unsigned char max_rgb = max(src_pixel.x,max(src_pixel.y,src_pixel.z));
        unsigned char min_rgb = min(src_pixel.x,min(src_pixel.y,src_pixel.z));

        mem[mem_pos].x = max_rgb - src_pixel.x + min_rgb;
        mem[mem_pos].y = max_rgb - src_pixel.y + min_rgb;
        mem[mem_pos].z = max_rgb - src_pixel.z + min_rgb;
    }
}

__global__
void copy_uchar_array_to_uchar4_dim(uchar4* mem, size_t mem_pitch,
                                    unsigned char* src,
                                    int width,int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if((gidx < width) && (gidy < height)){
        size_t mem_pos = gidx+gidy*mem_pitch/sizeof(uchar4);
        size_t src_pos = gidy*width+gidx;
        mem[mem_pos].x = src[4*src_pos+0];
        mem[mem_pos].y = src[4*src_pos+1];
        mem[mem_pos].z = src[4*src_pos+2];
        mem[mem_pos].w = src[4*src_pos+3];
    }
}

__global__
void copy_uchar4_dim_to_uchar_array(unsigned char* dst,
                                    uchar4* mem, size_t mem_pitch,
                                    int width,int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if((gidx < width) && (gidy < height)){
        size_t mem_pos = gidx+gidy*mem_pitch/sizeof(uchar4);
        size_t dst_pos = gidy*width+gidx;
        dst[4*dst_pos+0] = mem[mem_pos].x;
        dst[4*dst_pos+1] = mem[mem_pos].y;
        dst[4*dst_pos+2] = mem[mem_pos].z;
        dst[4*dst_pos+3] = mem[mem_pos].w;
    }
}

void rgb_invert_in_cuda(unsigned char* dst, unsigned char* src, int width, int height){
    uchar4* mem_uchar;
    size_t mem_pitch;
    unsigned char* uchar_buffer;
    dim3 blockDim(64,2);
    dim3 gridDim(divRoundUp(width, blockDim.x), divRoundUp(height, blockDim.y));

    getError(cudaMalloc(&uchar_buffer,4*sizeof(unsigned char)*width*height));
    getError(cudaMemcpy(uchar_buffer, src, 4*sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice));
    getError(cudaMallocPitch(&mem_uchar, &mem_pitch, width*sizeof(uchar4), height));

    copy_uchar_array_to_uchar4_dim<<<gridDim, blockDim>>>(mem_uchar, mem_pitch, uchar_buffer, width, height);
    rgb_invert_all<<<gridDim, blockDim>>>(mem_uchar, mem_pitch, width, height);
    copy_uchar4_dim_to_uchar_array<<<gridDim, blockDim>>>(uchar_buffer, mem_uchar, mem_pitch, width, height);

    getError(cudaMemcpy(dst, uchar_buffer, 4*sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));
    getError(cudaDeviceSynchronize());
}

__global__
void rgb_invert_all_uchar_array(unsigned char* mem, int width, int height){
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;

    if((gidx < width) && (gidy < height)){
        size_t pos = 4*(gidy*width+gidx);

        unsigned char max_rgb = max(mem[pos],max(mem[pos+1],mem[pos+2]));
        unsigned char min_rgb = min(mem[pos],min(mem[pos+1],mem[pos+2]));

        mem[pos]   = max_rgb - mem[pos]   + min_rgb;
        mem[pos+1] = max_rgb - mem[pos+1] + min_rgb;
        mem[pos+2] = max_rgb - mem[pos+2] + min_rgb;
    }
}

void rgb_invert_in_cuda_uchar_array(unsigned char* dst, unsigned char* src, int width, int height){
    unsigned char* uchar_buffer;
    dim3 blockDim(64,2);
    dim3 gridDim(divRoundUp(width, blockDim.x), divRoundUp(height, blockDim.y));

    getError(cudaMalloc(&uchar_buffer,4*sizeof(unsigned char)*width*height));
    getError(cudaMemcpy(uchar_buffer, src, 4*sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice));
    rgb_invert_all_uchar_array<<<gridDim, blockDim>>>(uchar_buffer, width, height);
    getError(cudaMemcpy(dst, uchar_buffer, 4*sizeof(unsigned char)*width*height, cudaMemcpyDeviceToHost));
    getError(cudaDeviceSynchronize());
}

// http://on-demand.gputechconf.com/gtc/2014/jp/sessions/4002.pdf
void rgb_invert_in_cpu(unsigned char* dst, unsigned char* src, int width, int height){
    for(int i=0;i<width*height;i++){
        unsigned char r = src[4*i+0];
        unsigned char g = src[4*i+1];
        unsigned char b = src[4*i+2];
        unsigned char min_rgb = min(r,min(g,b));
        unsigned char max_rgb = max(r,max(g,b));

        dst[4*i+0] = max_rgb - r + min_rgb;
        dst[4*i+1] = max_rgb - g + min_rgb;
        dst[4*i+2] = max_rgb - b + min_rgb;
        dst[4*i+3] = src[4*i+3];
    }
}