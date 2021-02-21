/* Doing 2D FFT in MKL using 1D vectors as data containers. */

#include <stdio.h>
#include <mkl.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>

using namespace std;

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


int main(){
    cout << "Init\n";
    MKL_LONG img_size[2] = {500, 500};
    long img_total = img_size[0] * img_size[1];
    int i_row, i_col;

    vector<complex<float>> img_dat(img_total);
    vector<complex<float>> f_img_dat(img_total);

    cout << "Preparing input...\n";
    for (int i = 0; i < img_total; i++){
        i_row = i / (int)img_size[1];
        i_col = i % (int)img_size[1];
        if (i_row >= 200 && i_row <= 300 && i_col >= 200 && i_col <= 300)
            img_dat[i] = 1.5;
        else
            img_dat[i] = 0.0;
    }

    output_txt<vector<complex<float>>>("in_mag.csv", img_dat, img_size);
    cout << "Input generated.\n";

    DFTI_DESCRIPTOR_HANDLE mkl_plan;
    MKL_LONG status;

    status = DftiCreateDescriptor(&mkl_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, img_size);
    DftiSetValue(mkl_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mkl_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(mkl_plan);
	// Call data(): returns a direct pointer to the memory array used internally by the vector to store its owned elements.
    DftiComputeForward(mkl_plan, img_dat.data(), f_img_dat.data());

    /* Image output */
    output_txt<vector<complex<float>>>("out_mag.csv", f_img_dat, img_size);

    DftiFreeDescriptor(&mkl_plan);
    return 0;
}