/* 2D FFT using CuFFT. */

#include <stdio.h>
#include <cufft.h>
#include <iostream>
#include <cstdlib>

using namespace std;
typedef float2 Complex;

void print_array_complex(Complex* arr, int ny, int nx)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            cout << arr[iy * nx + ix].x << " + " << arr[iy * nx + ix].y << "j ";
        }
        cout << endl;
    }
}

int main()
{
    int ny = 5;
    int nx = 5;
    int size = ny * nx;
    Complex *img, *res;
    cufftComplex *img_dev;

    img = new Complex[size];
    res = new Complex[size];
    for (int i = 0; i < size; i++)
    {
        //img[i].x = (float)rand() / (float)RAND_MAX;
        //img[i].y = (float)rand() / (float)RAND_MAX;
        img[i].x = 5.0;
        img[i].y = 0.0;
    }
    
    print_array_complex(img, ny, nx);
    cout << "=======================" << endl;
    
    cudaMalloc(&img_dev, sizeof(cufftComplex) * size);
    cudaMemcpy(img_dev, img, sizeof(cufftComplex) * size, cudaMemcpyHostToDevice);
    
    cufftHandle plan;
    cufftPlan2d(&plan, ny, nx, CUFFT_C2C);
    
    cufftExecC2C(plan, (cufftComplex *)img_dev, (cufftComplex *)img_dev, CUFFT_FORWARD);
    
    cudaMemcpy(res, img_dev, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
    
    print_array_complex(res, ny, nx);
    
    cufftDestroy(plan);
    cudaFree(img_dev);
}