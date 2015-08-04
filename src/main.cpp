// thanks to https://github.com/madsravn/easyCuda/blob/master/main.cpp

#include <iostream>
#include <cuda.h>
#include "lodepng.h"
#include <cuda_runtime.h>
#include "kernel.h"
#include <sys/time.h>
#include <functional>

long time_diff_us(struct timeval st, struct timeval et){
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}


int main(int argc, char** argv){
    if(argc != 3){
        std::cout << "usage:" << argv[0] << " src.png dst.png" << std::endl;
        return 1;
    }
    const char* input_file  = argv[1];
    const char* output_file = argv[2];
    std::vector<unsigned char> in_image;

    unsigned int width, height;
    const unsigned input_error = lodepng::decode(in_image, width, height, input_file);
    if(input_error){
        std::cout << "decoder error " << input_error << ": " << lodepng_error_text(input_error) << std::endl;
        return 1;
    }

    unsigned char* input_image  = new unsigned char[in_image.size()];
    unsigned char* output_image = new unsigned char[in_image.size()];
    std::copy(in_image.begin(), in_image.end(), input_image);

    struct timeval st;
    struct timeval et;
    gettimeofday(&st, NULL);

    
    rgb_invert_in_cuda(output_image, input_image, width, height);
    // rgb_invert_in_cpu(output_image, input_image, width, height);

    gettimeofday(&et, NULL);
    long us = time_diff_us(st, et);
    printf("%lu\n",us);

    std::vector<unsigned char> out_image(output_image, output_image+in_image.size());
    unsigned output_error = lodepng::encode(output_file, out_image, width, height);
    if(output_error){
        std::cout << "encoder error " << output_error << ": "<< lodepng_error_text(output_error) << std::endl;
        return 1;
    }

    return 0;
}
