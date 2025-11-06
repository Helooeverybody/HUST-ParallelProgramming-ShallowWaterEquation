#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// domain problem 
#define H  10.0
#define X  8.0
#define Y  8.0
#define time 3.0
#define g 9.81
#define delta_h 20.0 
#define initial_u 10.0
#define initial_v 10.0
#define delta_x 0.5
#define delta_y 0.5
#define CB_name 'T'
#define C 0.5

// Index 
#define INDEX(i, j, k) ((i)*((int)(Y/delta_y))*3 + (j)*3 + (k))

void Update_FluxF(double* U, double * FU, int sizex, int sizey){
    for (int i = 0; i < sizex ; i++){
        for (int j = 0; j < sizey;j++){
            double h = U[INDEX(i,j,0)];
            double v = U[INDEX(i,j,1)] / U[INDEX(i,j,0)];
            double u = U[INDEX(i,j,2)] / U[INDEX(i,j,0)];
    
            FU[INDEX(i,j,0)] = h*u; 
            FU[INDEX(i,j,1)] = h*u*u + 0.5f*g*h*h;
            FU[INDEX(i,j,2)] = h*u*v; 
        }
    }
}

void Update_FluxG(double * U, double *GU, int sizex, int sizey){
    for (int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey;j++){
            double h = U[INDEX(i,j,0)];
            double v = U[INDEX(i,j,1)]/ U[INDEX(i,j,0)];
            double u = U[INDEX(i,j,2)]/ U[INDEX(i,j,0)];

            GU[INDEX(i,j,0)] = h*v;
            GU[INDEX(i,j,1)] = h*u*v;
            GU[INDEX(i,j,2)] = h*v*v + 0.5*g*h*h;
        }
    }
}

double CNF_Condition(double* U, double sizex, double sizey, double * delta_t , double* alpha_u,double * alpha_v){
    // Update delta_t , alpha_u, alpha_v follow by CNF condition
    double max_speed_u = 0.0f;
    double max_speed_v = 0.0f;

    for (int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++){
            double h = U[INDEX(i,j,0)];
            double v = U[INDEX(i,j,1)]/U[INDEX(i,j,0)];
            double u = U[INDEX(i,j,2)]/U[INDEX(i,j,0)];

            if ((fabs(v) + sqrt(g*h)) > max_speed_v){
                max_speed_v = fabs(v) + sqrt(g*h);
            }
            if ((fabs(u) + sqrt(g*h)) > max_speed_u){
                max_speed_u = fabs(u) + sqrt(g*h);
            }
        }
    }
    *alpha_u = max_speed_u;
    *alpha_v = max_speed_v;
    *delta_t = C*fmin(delta_x/max_speed_u, delta_y/max_speed_v);
}

void InitialConditionBound(double *U, double* FU, double* GU, int sizex , int sizey){
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

    // update Flux F and Flux G matrix 
    Update_FluxF(U , FU, sizex , sizey);
    Update_FluxG(U, GU , sizex , sizey);
}

void Update_half_step_FluxF(double *U_half, double* FU_half){
    double h , u , v;
    h = U_half[0];
    v = U_half[1]/U_half[0];
    u = U_half[2]/U_half[0];

    FU_half[0] = h*u;
    FU_half[1] = h*u*u + 0.5f*g*h*h;
    FU_half[2] = h*u*v;
}

void Update_half_step_FluxG(double* U_half , double* GU_half){
    double h , u , v;
    h = U_half[0];
    v = U_half[1]/U_half[0];
    u = U_half[2]/U_half[0];

    GU_half[0] = h*v;
    GU_half[1] = h*u*v;
    GU_half[2] = h*v*v + 0.5*g*h*h;
}

void Update_U(double* U_new, double* U, double* FU , double* GU, int sizex , int sizey, double delta_t, double alpha_x , double alpha_y){

    for (int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++){
            
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
        
                U_half_up[k] = 0.5*(U[INDEX(i,j,k)] + U_up) - (delta_t/(2.0*delta_x))*(FU_up - FU[INDEX(i,j,k)]);
                U_half_down[k] = 0.5*(U[INDEX(i,j,k)] + U_down) - (delta_t/(2.0*delta_x))*(FU_down - FU[INDEX(i,j,k)]);
                U_half_left[k] = 0.5*(U[INDEX(i,j,k)] + U_left) - (delta_t/(2.0*delta_y))*(GU_left - GU[INDEX(i,j,k)]);
                U_half_right[k] = 0.5*(U[INDEX(i,j,k)] + U_right) - (delta_t/(2.0*delta_y))*(GU_right - GU[INDEX(i,j,k)]);

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
                U_new[INDEX(i,j,k)] = U[INDEX(i,j,k)] - (delta_t/delta_x)*(FU_half_up[k] - FU_half_down[k]) - (delta_t/delta_y)*(GU_half_right[k] - GU_half_left[k]) + alpha_x*0.002*U_entropy_x[k] + alpha_y*0.002*U_entroy_y[k];
            }
        }
    }
}


void store_matrix(FILE *fp , double * U, int step, int sizex , int sizey){
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

    int sizex = X / delta_x;
    int sizey = Y / delta_y;

    double *U = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *FU = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *GU = (double*)malloc(sizeof(double)*3*sizex*sizey);

    /* GOAL : SOLVING DIFFERENTIAL EQUATION 
            ∂U/∂t  + ∂F(U)/∂x + ∂G(U)/∂y = 0     
    */ 
    FILE *fp = fopen("shallow_water_simulation.json", "w");

    // store configuration 
    fprintf(fp, "{\n"); 
    fprintf(fp, "\"name\": \"lax wendroff finite difference\",\n");
    fprintf(fp, "\"H\": %f,\n", H); 
    fprintf(fp, "\"X\": %f,\n", X); 
    fprintf(fp, "\"Y\": %f,\n", Y); 
    fprintf(fp, "\"time\": %f,\n", time); 
    fprintf(fp, "\"delta_x\": %f,\n", delta_x); 
    fprintf(fp, "\"delta_y\": %f\n", delta_y); 
    
    // initial state
    InitialConditionBound(U,FU,GU, sizex, sizey);
    int step = 0; 
    double count_time = 0; 
    double delta_t, alpha_x , alpha_y;
    CNF_Condition(U,sizex, sizey, &delta_t, &alpha_x, &alpha_y); 

    // initial state store 
    store_matrix(fp , U, step, sizex, sizey);

    while (count_time < time){
        double* U_new = (double*)malloc(sizeof(double)*3*sizex*sizey);

        // calculus U half step 
        Update_U(U_new, U, FU, GU, sizex, sizey, delta_t, alpha_x,alpha_y);

        // update flux F and G with final U 
        Update_FluxF(U_new, FU, sizex, sizey);
        Update_FluxG(U_new, GU, sizex ,sizey);

        // update U 
        for (int i = 0 ; i < sizex; i++){
            for (int j = 0; j < sizey; j++){
                for(int k = 0; k < 3; k++){
                    U[INDEX(i,j,k)] = U_new[INDEX(i,j,k)];
                }
            }
        }

        step++;
        count_time += delta_t;

        // update delta_t 
        CNF_Condition(U, sizex, sizey, &delta_t, &alpha_x, &alpha_y);

        // store matrix state
        store_matrix(fp,U, step, sizex,sizey); 

        free(U_new);
    }

    // total step store 
    fprintf(fp,",\"total_step\": %d \n", step+1);
    // final store 
    fprintf(fp,"}");

    free(U);
    free(FU);
    free(GU);

    return 0;
}