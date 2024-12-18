#pragma once
#ifdef __INTELLISENSE__
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
    unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
    unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
#endif