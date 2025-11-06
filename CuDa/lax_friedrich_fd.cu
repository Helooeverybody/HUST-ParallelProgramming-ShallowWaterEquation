#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// domain problem 
#define H  3.0
#define X  8.0
#define Y  8.0
#define time 10.0
#define g 9.81
#define delta_h 5.0 
#define initial_u 20.0
#define initial_v 20.0 
#define delta_x 0.5  
#define delta_y 0.5
#define CB_name 'T'
#define C 0.9 

#define M ((int)(X/delta_x) * (int)(Y/delta_y))
#define GridSize_h 4
#define GridSize_w 4
#define BlockSize_h (int)((X/delta_x) / GridSize_h)
#define BlockSize_w (int)((Y/delta_y) / GridSize_w)


#define INDEX(i, j, k) ((i)*((int)(Y/delta_y))*3 + (j)*3 + (k))


__global__ void Update_FluxF(double* U, double* FU, int sizex, int sizey){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < sizex && j < sizey) {
        double h = U[INDEX(i,j,0)];
        double v = U[INDEX(i,j,1)] / U[INDEX(i,j,0)];
        double u = U[INDEX(i,j,2)] / U[INDEX(i,j,0)];
        
        FU[INDEX(i,j,0)] = h*u;
        FU[INDEX(i,j,1)] = h*u*u + 0.5f*g*h*h;
        FU[INDEX(i,j,2)] = h*u*v;
    }
    __syncthreads();
}

__global__ void Update_FluxG(double* U, double* GU, int sizex, int sizey){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < sizex && j < sizey) {
        double h = U[INDEX(i,j,0)];
        double v = U[INDEX(i,j,1)]/ U[INDEX(i,j,0)];
        double u = U[INDEX(i,j,2)]/ U[INDEX(i,j,0)];

        GU[INDEX(i,j,0)] = h*v;
        GU[INDEX(i,j,1)] = h*u*v;
        GU[INDEX(i,j,2)] = h*v*v + 0.5f*g*h*h;
    }
    __syncthreads();
}

__global__ void CNF_Condition(double* U,double* local_max_u, double* local_max_v,double sizex, double sizey){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = i * gridDim.y*blockDim.y + j;
    double speed_u;
    double speed_v;
    if (i < sizex && j < sizey) {
        double h = U[INDEX(i,j,0)];
        double v = U[INDEX(i,j,1)]/U[INDEX(i,j,0)];
        double u = U[INDEX(i,j,2)]/U[INDEX(i,j,0)];

        speed_v = fabs(v) + sqrt(g*h);
        speed_u = fabs(u) + sqrt(g*h);
       
    }
    local_max_u[idx] = speed_u;
    local_max_v[idx] = speed_v;
   
   
}
__global__ void ReductionMax(double *maxIn, double *maxOut){
    extern __shared__ double sdata[];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;


    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int idx = tx * gridDim.y*blockDim.y + ty;
    sdata[tid] = maxIn[idx];
    __syncthreads();

    for (int i = 1; i < blockDim.x*blockDim.y; i *= 2){
        if (tid % (2*i) == 0 && tid + i < blockDim.x*blockDim.y){
            sdata[tid] = fmax(sdata[tid] , sdata[tid + i]);
        }
        __syncthreads();
    }

    if (tid == 0) maxOut[blockIdx.x* gridDim.y + blockIdx.y] = sdata[0]; 
}

void InitialConditionBound(double* U, int sizex, int sizey) {
    for (int i = 0 ; i < sizex; i++){
        for (int j = 0; j < sizey; j++){
            // increase height in the left and right of a side 
            double h,v,u;
            if (i == 0){
                h = delta_h;
                v = initial_v;
                u = initial_u;
            }else if (i < (sizex/2)){
                h = ((double)(1/i))*delta_h;
                v = ((double)(1/i))*initial_v;
                u = ((double)(1/i))*initial_u;
            }else{
                h = 0;
                v = 0;
                u = 0;
            }
        U[INDEX(i,j,0)] = (H+h);
        U[INDEX(i,j,1)] = (H+h)*v;
        U[INDEX(i,j,2)] = (H+h)*u;

        }
    }
}
__global__ void Update_U(double* U_new, double* U, double* FU , double* GU,double delta_t, int sizex , int sizey){
   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double U_up, U_down , U_left, U_right;
    double FU_up , FU_down;
    double GU_left , GU_right;

    if (i < sizex && j < sizey) {
        for (int k = 0; k < 3; k++) {

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
            double Flux_u = (FU_up - FU_down);
            double Flux_v = (GU_right - GU_left);
            U_new[INDEX(i, j, k)] = 0.25f * (U_up + U_down + U_left + U_right)
                                          - (delta_t / (2.0f * delta_x)) * Flux_u
                                          - (delta_t / (2.0f * delta_y)) * Flux_v;
        }
    }
    __syncthreads();
}
void store_matrix(FILE* fp, double* U, int step, int sizex, int sizey) {
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
    fprintf(fp, "]\n");
}

int main(){

    int sizex = (int)(X / delta_x);
    int sizey = (int)(Y / delta_y);
    dim3 dimGrid(GridSize_h, GridSize_w);
    dim3 dimBlock(BlockSize_h, BlockSize_w);


    double *Ucpu = (double*)malloc(sizeof(double)* 3 * sizex * sizey);
    double *Ugpu, *UgpuNew, *FU, *GU;

    cudaMalloc((void**)&UgpuNew, sizeof(double)* 3 * sizex * sizey);
    cudaMalloc((void**)&Ugpu, sizeof(double)* 3 * sizex * sizey);
    cudaMalloc((void**)&FU, sizeof(double)* 3 * sizex * sizey);
    cudaMalloc((void**)&GU, sizeof(double)* 3 * sizex * sizey);
    ///////////////
    InitialConditionBound(Ucpu, sizex, sizey);
    cudaMemcpy(Ugpu, Ucpu, sizeof(double)* 3 * sizex * sizey, cudaMemcpyHostToDevice);
    Update_FluxF<<<dimGrid,dimBlock>>>(Ugpu,FU, sizex, sizey);
    Update_FluxG<<<dimGrid,dimBlock>>>(Ugpu,GU, sizex, sizey);
    ///////////
    FILE *fp = fopen("shallow_water_simulation.json", "w");
    if (fp==NULL){
        printf("Error");
    }

    // store configuration 
    fprintf(fp, "{\n");
    fprintf(fp, "\"name\": \"lax friedrichs finite difference\",\n");
    fprintf(fp, "\"H\": %f,\n", H);
    fprintf(fp, "\"X\": %f,\n", X);
    fprintf(fp, "\"Y\": %f,\n", Y);
    fprintf(fp, "\"time\": %f,\n", time);
    fprintf(fp, "\"delta_x\": %f,\n", delta_x);
    fprintf(fp, "\"delta_y\": %f\n", delta_y);
    

    int step = 0;
    double count_time = 0;


    double *localMaxu, *localMaxv;
    cudaMalloc((void **)&localMaxu, M*sizeof(double));
    cudaMalloc((void **)&localMaxv, M*sizeof(double));
     
    double *blockMaxu, *blockMaxv;   
    double *globalMaxu, *globalMaxv;  
    cudaMalloc((void**)&globalMaxu, 1*sizeof(double));
    cudaMalloc((void**)&globalMaxv, 1*sizeof(double));
    cudaMalloc((void**)&blockMaxu, BlockSize_h*BlockSize_w*sizeof(double));
    cudaMalloc((void**)&blockMaxv, BlockSize_h*BlockSize_w*sizeof(double));

    double max_u, max_v;
    
   
    while (count_time < time){
        
      
        CNF_Condition<<<dimGrid, dimBlock>>>(Ugpu, localMaxu, localMaxv, sizex, sizey);
        cudaDeviceSynchronize();
        
        ReductionMax<<<dimGrid , dimBlock, BlockSize_h*BlockSize_w*sizeof(double)>>>(localMaxu,blockMaxu); 
        ReductionMax<<<1,GridSize_h*GridSize_w, GridSize_h*GridSize_w*sizeof(double)>>>(blockMaxu,globalMaxu); 
        ReductionMax<<<dimGrid , dimBlock, BlockSize_h*BlockSize_w*sizeof(double)>>>(localMaxv,blockMaxv); 
        ReductionMax<<<1,GridSize_h*GridSize_w, GridSize_h*GridSize_w*sizeof(double)>>>(blockMaxv,globalMaxv); 
        
        cudaMemcpy(&max_u, globalMaxu, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&max_v, globalMaxv, sizeof(double), cudaMemcpyDeviceToHost);
       
        double delta_t = C*fmin(delta_x/max_u, delta_y/max_v);

        cudaDeviceSynchronize();
        
        Update_U<<<dimGrid, dimBlock>>>(UgpuNew, Ugpu, FU, GU, delta_t, sizex, sizey);
        cudaDeviceSynchronize();
        std::swap(Ugpu, UgpuNew);
        cudaDeviceSynchronize();
        Update_FluxF<<<dimGrid,dimBlock>>>(Ugpu,FU, sizex, sizey);
        Update_FluxG<<<dimGrid,dimBlock>>>(Ugpu,GU, sizex, sizey);
        cudaDeviceSynchronize();
        count_time += delta_t;
        cudaMemcpy(Ucpu, Ugpu, sizeof(double) * 3 * sizex * sizey, cudaMemcpyDeviceToHost);
        store_matrix(fp, Ucpu, step, sizex, sizey);
        step++;
        cudaMemcpy(Ugpu, Ucpu, sizeof(double)* 3 * sizex * sizey, cudaMemcpyHostToDevice);
        

    }
  
    fprintf(fp,",\"total_step\": %d \n", step);
    // final store 
    fprintf(fp,"}");
    fclose(fp);
    cudaFree(Ugpu);
    cudaFree(FU);
    cudaFree(GU);
    cudaFree(UgpuNew);
    free(Ucpu);

    return 0;
}


