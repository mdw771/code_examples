/* Doing 2D FFT in MKL using 2D arrays as data containers. */

#include <stdio.h>
#include <mkl.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <fstream>
#include <cmath>
#include <string>

using namespace std;

// Here we use a typename T as template parameter. If no template parameter is specified
// when the function is called, the compiler automatically deduce the data type from the
// argumment and fill in the data type. In this way we don't need to care about how the 2D
// array is defined (either complex img[ny][nx], or complex* img[ny], or complex** img) when
// writing the function signature. 
template<typename T>
void output_txt_2d(string fname, T img, MKL_LONG img_size[]){
    ofstream f_out;
    f_out.open(fname);
    for (int i = 0; i < img_size[0]; i++){
        for (int j = 0; j < img_size[1]; j++){
            f_out << abs(img[i][j]) << ",";
        }
        f_out << endl;
    }
    f_out.close();
    return;
}

int main(){
    cout << "Init\n";
    const int n_y = 500;
    const int n_x = 500;
    MKL_LONG img_size[2] = {n_y, n_x};
    long img_total = n_y * n_x;
    int i_row, i_col;

    complex<float> img_dat[n_y][n_x];
    complex<float> f_img_dat[n_y][n_x];
	
	// Generate input data: a rectangular aperture.
    for (int i = 0; i < img_total; i++){
        i_row = (long)i / img_size[1];
        i_col = i % img_size[1];
        if (i_row >= 200 && i_row <= 300 && i_col >= 200 && i_col <= 300)
            img_dat[i_row][i_col] = 1.5;
        else
            img_dat[i_row][i_col] = 0.0;
    }

    DFTI_DESCRIPTOR_HANDLE mkl_plan;
    MKL_LONG status;

    status = DftiCreateDescriptor(&mkl_plan, DFTI_SINGLE, DFTI_COMPLEX, 2, img_size);
    DftiSetValue(mkl_plan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mkl_plan, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(mkl_plan);
    DftiComputeForward(mkl_plan, img_dat, f_img_dat);

    /* Image output */
    output_txt_2d("in_mag.csv", img_dat, img_size);
    output_txt_2d("out_mag.csv", f_img_dat, img_size);

    DftiFreeDescriptor(&mkl_plan);
    return 0;
}
