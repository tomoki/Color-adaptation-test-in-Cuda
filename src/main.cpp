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
    if(argc < 3){
        std::cout << "usage:" << argv[0] << " src.png dst.png" << std::endl;
        return 1;
    }
    int mode = 0;  // for cuda
    if(argc == 4){ // this is silly implementation
        char* m = argv[3];
        mode = m[0] - '0';
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

    // first CUDA call takes a whlie.
    //   For benchmark, I do not take account of first call.
    const int N = 5;
    long sum_of_n_minus_one = 0;
    for(int i=0;i<N;i++){
        struct timeval st;
        struct timeval et;
        gettimeofday(&st, NULL);
        if(mode == 0){
            rgb_invert_in_cuda(output_image, input_image, width, height);
        }else if(mode == 1){
            rgb_invert_in_cpu(output_image, input_image, width, height);
        }else if(mode == 2){
            rgb_invert_in_cuda_uchar_array(output_image, input_image, width, height);
        }else{
            std::cout << "ERROR: unknown mode: " << mode << std::endl;
            return 1;
        }
        gettimeofday(&et, NULL);
        long us = time_diff_us(st, et);
        if(i != 0){
            sum_of_n_minus_one += us;
        }
    }
    printf("%lu\n",sum_of_n_minus_one/(N-1));

    std::vector<unsigned char> out_image(output_image, output_image+in_image.size());
    unsigned output_error = lodepng::encode(output_file, out_image, width, height);
    if(output_error){
        std::cout << "encoder error " << output_error << ": "<< lodepng_error_text(output_error) << std::endl;
        return 1;
    }

    return 0;
}
