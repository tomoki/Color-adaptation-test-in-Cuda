#ifndef KERNEL_H
#define KERNEL_H

void rgb_invert_in_cuda(unsigned char* dst, unsigned char* src, int width, int height);
void rgb_invert_in_cpu(unsigned char* dst, unsigned char* src, int width, int height);

#endif
