/* 
A Fresnel propagation (with Fresnel approximation) function implemented with MKL.
It also comes with a type traits which automatically deduces the underlying real types
from the complex vector type supplied. 
With Intel C++ compiler, compile using the following command: 
icpc -mkl -std=c++11 fresnel_propagation.cpp -o fresnel_propagation 
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <mkl.h>
#include <cmath>
#include <complex>
#include <tuple>

using namespace std;

template<typename T>
struct comp2real_tts{
    using real_arr_type = T;
};

template<typename T>
struct comp2real_tts<vector<complex<T> > >{
    using real_arr_type = vector<T>;
    using real_type = T;
};

template<typename T>
void output_txt(string fname, T &img, MKL_LONG img_size[]){
    ofstream f_out;
    f_out.open(fname);
    for (int i_row = 0; i_row < img_size[0]; i_row++){
        for (int i_col = 0; i_col < img_size[1]; i_col++){
            f_out << abs(img[i_row * (int)img_size[1] + i_col]) << ",";
        }
        f_out << endl;
    }
    f_out.close();
    return;
}

template<typename real_arr_type>
real_arr_type fftfreq(int n){
    real_arr_type arr(n);
    int j = 0;
    if (n % 2 == 0){
        for (int i = 0; i < n / 2; i++)
            arr[i] = float(i) / n;
        for (int i = n / 2; i < n; i++){
            arr[i] = (-n / 2 + j) / float(n);
            j++;
        }
    }
    else{
        for (int i = 0; i < (n - 1) / 2 + 1; i++)
            arr[i] = float(i) / n;
        for (int i = (n - 1) / 2; i < n; i++){
            arr[i] = (-(n - 1) / 2 + j) / float(n);
            j++;
        }
    }
    return arr;
}

template<typename real_arr_type>
tuple<real_arr_type, real_arr_type> gen_freq_mesh(float voxel_nm[], long shape[]){
    real_arr_type uu(shape[0] * shape[1]);
    real_arr_type vv(shape[0] * shape[1]);
    real_arr_type u = fftfreq<real_arr_type>(shape[0]);
    real_arr_type v = fftfreq<real_arr_type>(shape[1]);
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++){
            uu[i * shape[1] + j] = u[i] / voxel_nm[0];
            vv[i * shape[1] + j] = v[j] / voxel_nm[1];
        }
    }
    return tuple<real_arr_type, real_arr_type>(uu, vv);
}

template<typename complex_arr_type>
void get_kernel(float dist_nm, float lambda_nm, float voxel_nm[], long shape[], complex_arr_type& kernel){
    int size = shape[0] * shape[1];
    // Get the underlying real types from type traits according to the complex vector type.
    using real_arr_type = typename comp2real_tts<complex_arr_type>::real_arr_type;
    using real_type = typename comp2real_tts<complex_arr_type>::real_type;

    real_arr_type uu(size), vv(size);
    tuple<real_arr_type, real_arr_type> ret = gen_freq_mesh<real_arr_type>(voxel_nm, shape);
    uu = get<0>(ret);
    vv = get<1>(ret);
    for (int i = 0; i < size; i++)
    {
        complex<real_type> temp = complex<real_type>{0., -3.14159 * lambda_nm * dist_nm * (uu[i]*uu[i] + vv[i]*vv[i])};
        temp = exp(temp);
        kernel[i] = temp;
    }
}

template<typename complex_arr_type>
void fft2_inplace(complex_arr_type& arr, MKL_LONG img_shape[], bool norm=true){
    DFTI_DESCRIPTOR_HANDLE mkl_plan;
    MKL_LONG status;
    
    status = DftiCreateDescriptor(&mkl_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, img_shape);
    DftiSetValue(mkl_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mkl_plan, DFTI_PLACEMENT, DFTI_INPLACE);
    if (norm) DftiSetValue(mkl_plan, DFTI_FORWARD_SCALE, 1.0 / sqrt(img_shape[0] * img_shape[1]));
    DftiCommitDescriptor(mkl_plan);
    DftiComputeForward(mkl_plan, arr.data());

    DftiFreeDescriptor(&mkl_plan);
}

template<typename complex_arr_type>
void ifft2_inplace(complex_arr_type& arr, MKL_LONG img_shape[], bool norm=true){
    DFTI_DESCRIPTOR_HANDLE mkl_plan;
    MKL_LONG status;

    status = DftiCreateDescriptor(&mkl_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, img_shape);
    DftiSetValue(mkl_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mkl_plan, DFTI_PLACEMENT, DFTI_INPLACE);
    if (norm) DftiSetValue(mkl_plan, DFTI_BACKWARD_SCALE, 1.0 / sqrt(img_shape[0] * img_shape[1]));
    DftiCommitDescriptor(mkl_plan);
    DftiComputeBackward(mkl_plan, arr.data());

    DftiFreeDescriptor(&mkl_plan);
}

template<typename complex_arr_type>
void fresnel_propagate(complex_arr_type& wavefield, float dist_nm, float lambda_nm, float voxel_nm[], long shape[]){
    long wave_size = shape[0] * shape[1];
    complex_arr_type kernel(wave_size);
    get_kernel(dist_nm, lambda_nm, voxel_nm, shape, kernel);
    fft2_inplace(wavefield, shape);
    for (int i = 0; i < wave_size; i++){
        wavefield[i] = wavefield[i] * kernel[i];
    }
    ifft2_inplace(wavefield, shape);
}


int main(){
    cout << "Init\n";
    MKL_LONG img_shape[2] = {500, 500};
    long img_total = img_shape[0] * img_shape[1];
    int i_row, i_col;

    vector<complex<float> > wavefield(img_total);
    cout << "Preparing input...\n";
    for (int i = 0; i < img_total; i++){
        i_row = i / (int)img_shape[1];
        i_col = i % (int)img_shape[1];
        if (i_row >= 200 && i_row <= 300 && i_col >= 200 && i_col <= 300)
            wavefield[i] = 1.0;
        else
            wavefield[i] = 0.0;
    }

    float lambda_nm = 1.24 / 10;
    float dist_nm = 50000.;
    float voxel_nm[2] = {10., 10.};

    fresnel_propagate(wavefield, dist_nm, lambda_nm, voxel_nm, img_shape);

    output_txt<vector<complex<float> > >("propagated_mag.csv", wavefield, img_shape);

    return 0;
}
