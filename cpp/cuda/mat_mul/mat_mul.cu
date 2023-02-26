/* CUDA implementation of matrix multiplication. */

#include <iostream>
#include <math.h>
#include <stdio.h>

void __global__ mat_mul(int n_rows, int n_cols, float *x, float *y, float *res){
    /* n_rows, n_cols are dimensions of x. 
       Output dimension (res) will be n_row * n_row. */
    int row_st = blockIdx.x * gridDim.x + threadIdx.x;
    int col_st = blockIdx.y * gridDim.y + threadIdx.y;
    int n_thds_x = gridDim.x * blockDim.x;
    int n_thds_y = gridDim.y * blockDim.y;
    int ind_res_flat, ind_x_flat, ind_y_flat;
    float temp;
    for (int i_row = row_st; i_row < n_rows; i_row += n_thds_x){
        for (int i_col = col_st; i_col < n_cols; i_col += n_thds_y){
            ind_res_flat = i_row * n_cols + i_col;
            temp = 0;
            for (int i = 0; i < n_cols; i++){
                ind_x_flat = i_row * n_cols + i;
                ind_y_flat = i * n_rows + i_col;
                temp += (x[ind_x_flat] * y[ind_y_flat]);
            }
            res[ind_res_flat] = temp;
        }
    }
}

int print_mat(float *z, int n_rows, int n_cols)
{
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            printf("%f ", z[i * n_cols + j]);
        }
        printf("\n");
    }
    return 0;
}

int main(){
    int n_rows = 3;
    int n_cols = 3;
    float *x, *y, *res;
    float *x_dev, *y_dev, *res_dev;
    dim3 block_dim(1, 1);
    dim3 thread_dim(n_rows + 3, n_rows + 3);
    int mat_size = n_rows * n_cols * sizeof(float);
    
    x = (float*)malloc(mat_size);
    y = (float*)malloc(mat_size);
    res = (float*)malloc(mat_size);
    
    for (int i = 0; i < n_rows * n_cols; i++)
        x[i] = float(i);
        
    for (int i = 0; i < n_rows * n_cols; i++)
        y[i] = float(i);
        
    for (int i = 0; i < n_rows * n_cols; i++)
        res[i] = 0.;
        
    print_mat(x, n_rows, n_cols);
    std::cout << '*' << std::endl;
    print_mat(y, n_rows, n_cols);
    std::cout << '=' << std::endl;
        
    cudaMalloc(&x_dev, mat_size);
    cudaMalloc(&y_dev, mat_size);
    cudaMalloc(&res_dev, mat_size);
    
    cudaMemcpy(x_dev, x, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(res_dev, res, mat_size, cudaMemcpyHostToDevice);
    
    mat_mul<<<block_dim, thread_dim>>>(n_rows, n_cols, x_dev, y_dev, res_dev);
    
    cudaMemcpy(res, res_dev, mat_size, cudaMemcpyDeviceToHost);
    
    print_mat(res, n_rows, n_cols);
}
    