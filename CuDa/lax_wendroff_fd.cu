#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// domain problem 
#define H  10.0
#define X  8.0
#define Y  8.0
#define time 3.0
#define g 9.81
#define delta_h 20.0 
#define initial_u 4.0
#define initial_v 4.0
#define delta_x 0.5
#define delta_y 0.5
#define sizex  (int)(X/delta_x)
#define sizey  (int)(Y/delta_y)
#define CB_name 'T'
#define C 0.5
// Cuda size 
#define M sizex*sizey
#define GridSize_x 2
#define GridSize_y 2
#define BlockSize_x (int)(sizex/GridSize_x)
#define BlockSize_y (int)(sizey/GridSize_y)


// Index 
#define INDEX(i, j, k) ((i)*(sizey)*3 + (j)*3 + (k))

void InitialConditionBound(double *U){
    // initial boundary condition 
    for (int i = 0 ; i < sizex; i++){
        for (int j = 0; j < sizey; j++){
            // increase height in the left and right of a side 
            double h,v,u;
            if (i == 0){
                h = delta_h;
                v = initial_v;
                u = initial_u;
            }else{
                h = ((double)(1/(double)(i)))*delta_h;
                v = ((double)(1/(double)(i)))*initial_v;
                u = ((double)(1/(double)(i)))*initial_u;
            }
            U[INDEX(i,j,0)] = (H+h);
            U[INDEX(i,j,1)] = (H+h)*v;
            U[INDEX(i,j,2)] = (H+h)*u;

        }
    }
}

__global__ void Update_FluxF(double* U, double* FU){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < sizex && j < sizey){
        double h = U[INDEX(i,j,0)];
        double v = U[INDEX(i,j,1)]/h;
        double u = U[INDEX(i,j,2)]/h;

        FU[INDEX(i,j,0)] = h*u;
        FU[INDEX(i,j,1)] = h*u*u + 0.5f*g*h*h;
        FU[INDEX(i,j,2)] = h*u*v;
    }
    __syncthreads();
}

__global__ void Update_FluxG(double* U , double* GU ){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < sizex && j < sizey){

    double h = U[INDEX(i,j,0)];
    double v = U[INDEX(i,j,1)]/h;
    double u = U[INDEX(i,j,2)]/h;

    GU[INDEX(i,j,0)] = h*v;
    GU[INDEX(i,j,1)] = h*u*v;
    GU[INDEX(i,j,2)] = h*v*v + 0.5*g*h*h;
    }
    __syncthreads();
}

__global__ void CNF_Condition(double* delta_t, double* max_speed_u, double* max_speed_v){

    *delta_t = C*fmin(delta_x/ *max_speed_u, delta_y/ *max_speed_v) + 1e-6;
}

__global__ void Speed_Calculus(double *U, double *matrix_u, double *matrix_v){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i < sizex && j < sizey){
            
        double h = U[INDEX(i,j,0)];
        double v = U[INDEX(i,j,1)]/h; 
        double u = U[INDEX(i,j,2)]/h; 

        matrix_v[i*sizey + j] = fabs(v) + sqrt(g*h);
        matrix_u[i*sizey + j] = fabs(u) + sqrt(g*h);
    }
    __syncthreads();
}

__global__ void ReductionMax(double *Input, double *Output){
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x + tid;

    sdata[tid] = Input[idx];

    __syncthreads();

    for (int i = 1; i < blockDim.x; i *=2){          
        if (tid % 2*i == 0 && tid + i < blockDim.x){
            sdata[tid] = fmax(sdata[tid], sdata[tid + 1]);
        }
        __syncthreads();
    }

    if (tid == 0) Output[blockIdx.x] = sdata[0];
}

__device__ void Update_half_step_FluxF(double *U_half, double* FU_half){
    double h , u , v;
    h = U_half[0];
    v = U_half[1]/U_half[0];
    u = U_half[2]/U_half[0];

    FU_half[0] = h*u;
    FU_half[1] = h*u*u + 0.5f*g*h*h;
    FU_half[2] = h*u*v;
}

__device__ void Update_half_step_FluxG(double* U_half , double* GU_half){
    double h , u , v;
    h = U_half[0];
    v = U_half[1]/U_half[0];
    u = U_half[2]/U_half[0];

    GU_half[0] = h*v;
    GU_half[1] = h*u*v;
    GU_half[2] = h*v*v + 0.5*g*h*h;
}
__global__ void Update_U(double* U_new, double* U, double* FU , double* GU, double* delta_t, double *alpha_x , double *alpha_y){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    
    if (i < sizex && j < sizey){
        double U_half_up[3];
        double U_half_down[3];
        double U_half_left[3];
        double U_half_right[3]; 
        double U_entropy_x[3];
        double U_entroy_y[3];

        for (int k = 0; k < 3; k++){
            double U_up, U_down , U_left, U_right;
            double FU_up , FU_down; 
            double GU_left , GU_right; 

            // boundary condition  
            // U_up 
            if (i == sizex - 1) { 
                if (CB_name == 'R' && k == 2)
                    U_up = -U[INDEX(i,j,k)];
                else
                    U_up = U[INDEX(i,j,k)];
            } else {
                U_up = U[INDEX(i+1,j,k)];
            }
            // U_down 
            if (i == 0) {
                if (CB_name == 'R' && k == 2)
                    U_down = -U[INDEX(i,j,k)];
                else
                    U_down = U[INDEX(i,j,k)];
            } else {
                U_down = U[INDEX(i-1,j,k)];
            }
                
            // U_left
            if (j == 0) {
                if (CB_name == 'R' && k == 1)
                    U_left = -U[INDEX(i,j,k)];
                else
                    U_left = U[INDEX(i,j,k)];
            } else {
                U_left = U[INDEX(i,j-1,k)];
            }
                
            // U_right 
            if (j == sizey - 1) {
                if (CB_name == 'R' && k == 1)
                    U_right = -U[INDEX(i,j,k)];
                else
                    U_right = U[INDEX(i,j,k)];
            } else {
                U_right = U[INDEX(i,j+1,k)];
            }
                
            // FU_up 
            if (i == sizex - 1) {
                if (CB_name == 'R' && k == 0){
                    FU_up = -FU[INDEX(i,j,k)];
                }else if (CB_name == 'R' && k == 2){
                    FU_up = -FU[INDEX(i,j,k)];
                }else{
                    FU_up = FU[INDEX(i,j,k)];
                }
            } else {
                FU_up = FU[INDEX(i+1,j,k)];
            }
                
            // FU_down 
            if (i == 0) {
                if (CB_name == 'R' && k == 0){
                    FU_down = -FU[INDEX(i,j,k)];
                }else if (CB_name == 'R' && k == 2) {
                    FU_down = -FU[INDEX(i,j,k)];
                }else{
                    FU_down = FU[INDEX(i,j,k)];
                }
            } else {
                FU_down = FU[INDEX(i-1,j,k)];
            }
                
            // GU_left 
            if (j == 0) {
                if (CB_name == 'R' && k == 0){
                    GU_left = -GU[INDEX(i,j,k)];
                }else if (CB_name == 'R' && k == 1){
                    GU_left = -GU[INDEX(i,j,k)];
                }else{
                    GU_left = GU[INDEX(i,j,k)];
                }
            } else {
                GU_left = GU[INDEX(i,j-1,k)];
            }

            // GU_right 
            if (j == sizey - 1) {
                if (CB_name == 'R' && k == 0){
                    GU_right = -GU[INDEX(i,j,k)];
                }else if (CB_name == 'R' && k == 1){
                    GU_right = -GU[INDEX(i,j,k)];
                }else{
                GU_right = GU[INDEX(i,j,k)];
                }
            } else {
                GU_right = GU[INDEX(i,j+1,k)];
            }
        
            U_half_up[k] = 0.5*(U[INDEX(i,j,k)] + U_up) - (*delta_t/(2.0*delta_x))*(FU_up - FU[INDEX(i,j,k)]);
            U_half_down[k] = 0.5*(U[INDEX(i,j,k)] + U_down) - (*delta_t/(2.0*delta_x))*(FU_down - FU[INDEX(i,j,k)]);
            U_half_left[k] = 0.5*(U[INDEX(i,j,k)] + U_left) - (*delta_t/(2.0*delta_y))*(GU_left - GU[INDEX(i,j,k)]);
            U_half_right[k] = 0.5*(U[INDEX(i,j,k)] + U_right) - (*delta_t/(2.0*delta_y))*(GU_right - GU[INDEX(i,j,k)]);

            // update entropy 
            U_entropy_x[k] = (U_up - 2*U[INDEX(i,j,k)] + U_down);
            U_entroy_y[k] = (U_right - 2*U[INDEX(i,j,k)] + U_left);
        }

        // update FU and GU half step 
        double FU_half_up[3];
        double FU_half_down[3];
        double GU_half_left[3];
        double GU_half_right[3];

        Update_half_step_FluxF(U_half_up, FU_half_up);
        Update_half_step_FluxF(U_half_down, FU_half_down);
        Update_half_step_FluxG(U_half_left, GU_half_left);
        Update_half_step_FluxG(U_half_right, GU_half_right);
            
        // update U
        for (int k = 0; k < 3; k++){
            U_new[INDEX(i,j,k)] = U[INDEX(i,j,k)] - (*delta_t/delta_x)*(FU_half_up[k] - FU_half_down[k]) - (*delta_t/delta_y)*(GU_half_right[k] - GU_half_left[k]) + (*alpha_x)*0.002*U_entropy_x[k] + (*alpha_y)*0.002*U_entroy_y[k];
        }
    }
    __syncthreads();
}

void store_matrix(FILE *fp , double * U, int step){
    fprintf(fp, ",\"step_%d\": [\n", step);
    for (int i = 0; i < sizex; i++) {
        fprintf(fp, "  [");
        for (int j = 0; j < sizey; j++) {
            int idx = (i * sizey + j) * 3;
            double x = U[idx];
            double y = U[idx + 1];
            double z = U[idx + 2];
            fprintf(fp, "[%f, %f, %f]", x, y, z);
            if (j < sizey - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "]");
        if (i < sizex - 1) fprintf(fp, ",\n");
        else fprintf(fp, "\n");
    }
    fprintf(fp , "]\n");
}


int main(){
    double *U = (double*)malloc(sizeof(double)*3*sizex*sizey);

    double *U_gpu, *FU_gpu, *GU_gpu;
    cudaMalloc((void **)&U_gpu,sizeof(double)*sizex*sizey*3);
    cudaMalloc((void **)&FU_gpu,sizeof(double)*sizex*sizey*3);
    cudaMalloc((void **)&GU_gpu,sizeof(double)*sizex*sizey*3);

    /* GOAL : SOLVING DIFFERENTIAL EQUATION      
            ∂U/∂t  + ∂F(U)/∂x + ∂G(U)/∂y = 0     
    */ 

    FILE *fp = fopen("shallow_water_simulation.json", "w");
    // store configuration 
    fprintf(fp, "{\n"); 
    fprintf(fp, "\"H\": %f,\n", H); 
    fprintf(fp, "\"X\": %f,\n", X); 
    fprintf(fp, "\"Y\": %f,\n", Y); 
    fprintf(fp, "\"time\": %f,\n", time); 
    fprintf(fp, "\"delta_x\": %f,\n", delta_x); 
    fprintf(fp, "\"delta_y\": %f\n", delta_y);

    // initial state
    InitialConditionBound(U);

    cudaMemcpy(U_gpu,U,sizeof(double)*sizex*sizey*3,cudaMemcpyHostToDevice);

    dim3 dimGrid(GridSize_x, GridSize_y);
    dim3 dimBlock(BlockSize_x,BlockSize_y);

    Update_FluxF<<<dimGrid,dimBlock>>>(U_gpu , FU_gpu);
    Update_FluxG<<<dimGrid,dimBlock>>>(U_gpu , GU_gpu);

    int step = 0;
    double count_time = 0;
    // initial state store 
    store_matrix(fp , U, step);
    double delta_t_cpu;
    double *delta_t_gpu;

    double *max_speed_u, *max_speed_v;
    cudaMalloc((void **)&max_speed_u,sizeof(double));
    cudaMalloc((void **)&max_speed_v,sizeof(double));
    cudaMalloc((void **)&delta_t_gpu,sizeof(double));

    
    while (count_time < time){
        // copy state to gpu 
        cudaMemcpy(U_gpu,U,sizeof(double)*sizex*sizey*3,cudaMemcpyHostToDevice);
        double *matrix_u, *matrix_v, *matrix_u_out, *matrix_v_out;
        cudaMalloc((void **)&matrix_u, sizeof(double)*sizex*sizey);
        cudaMalloc((void **)&matrix_v, sizeof(double)*sizex*sizey);
        cudaMalloc((void **)&matrix_u_out,sizeof(double)*GridSize_x*GridSize_y);
        cudaMalloc((void **)&matrix_v_out,sizeof(double)*GridSize_x*GridSize_y);
        Speed_Calculus<<<dimGrid, dimBlock>>>(U_gpu,matrix_u,matrix_v);

        // update delta_t 
        ReductionMax<<<GridSize_x*GridSize_y,BlockSize_x*BlockSize_y,sizeof(double)*BlockSize_x*BlockSize_y>>>(matrix_u,matrix_u_out);
        ReductionMax<<<GridSize_x*GridSize_y,BlockSize_x*BlockSize_y,sizeof(double)*BlockSize_x*BlockSize_y>>>(matrix_v,matrix_v_out);
        
        ReductionMax<<<1,GridSize_x*GridSize_y,sizeof(double)*GridSize_x*GridSize_y>>>(matrix_u_out,max_speed_u);
        ReductionMax<<<1,GridSize_x*GridSize_y,sizeof(double)*GridSize_x*GridSize_y>>>(matrix_v_out,max_speed_v);

        CNF_Condition<<<1,1>>>(delta_t_gpu,max_speed_u,max_speed_v);

        // update new state 
        double *U_new_gpu;
        cudaMalloc((void **)&U_new_gpu, sizeof(double)*sizex*sizey*3);

        Update_U<<<dimGrid, dimBlock>>>(U_new_gpu,U_gpu,FU_gpu, GU_gpu, delta_t_gpu, max_speed_u, max_speed_v);

        Update_FluxF<<<dimGrid, dimBlock>>>(U_new_gpu, FU_gpu);
        Update_FluxG<<<dimGrid, dimBlock>>>(U_new_gpu, GU_gpu);

        cudaMemcpy(U,U_new_gpu,sizeof(double)*sizey*sizex*3, cudaMemcpyDeviceToHost);

        cudaMemcpy(&delta_t_cpu,delta_t_gpu,sizeof(double), cudaMemcpyDeviceToHost);
        count_time += delta_t_cpu;
        printf("%f \n",delta_t_cpu);
        step += 1;
        store_matrix(fp , U, step);

        cudaFree(U_new_gpu);
        cudaFree(matrix_u);
        cudaFree(matrix_v);
        cudaFree(matrix_u_out);
        cudaFree(matrix_v_out);
    }  

    // total step store 
    fprintf(fp,",\"total_step\": %d \n", step+1);
    // final store 
    fprintf(fp,"}");

    free(U);
    cudaFree(U_gpu);
    cudaFree(FU_gpu);
    cudaFree(GU_gpu);
    return 0;
}



