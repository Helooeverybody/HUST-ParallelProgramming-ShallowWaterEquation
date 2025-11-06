#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// domain problem 
#define H  3.0
#define X  20.0
#define Y  20.0
#define time 5.0
#define g 9.81
#define delta_h 20.0 
#define initial_u 10.0
#define initial_v 10.0
#define delta_x 2.0
#define delta_y 2.0
#define CB_name 'R'
#define C 0.1

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
            GU[INDEX(i,j,2)] = h*v*v + 0.5f*g*h*h;
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
                h = ((double)(1/i))*delta_h;
                v = ((double)(1/i))*initial_v;
                u = ((double)(1/i))*initial_u;
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


double Update_Calculus(double U_central, double FU_central, double GU_central , 
                    double U_up, double U_down, double U_left, double U_right,
                    double FU_up , double FU_down , double GU_left, double GU_right,
                    double alpha_u, double alpha_v, double delta_t){

    double Flux_margin_up = 0.5f*(FU_central - FU_up); 
    double Flux_margin_down = 0.5f*(-FU_central + FU_down);
    double Flux_margin_left = 0.5f*(-GU_central + GU_left);
    double Flux_margin_right = 0.5f*(GU_central - GU_right);

    double Flux_u = (Flux_margin_down - Flux_margin_up);
    double Flux_v = (Flux_margin_left - Flux_margin_right);

    double final_value = 0.25f*(U_down + U_left + U_up + U_right) - (delta_t/2.0f*delta_x)*Flux_u - (delta_t/2.0f*delta_y)*Flux_v;

    return final_value;
}


void Update_U(double* U_new, double* U, double* FU , double* GU, int sizex , int sizey, double delta_t , double alpha_u , double alpha_v){
    for (int i = 0; i < sizex; i++){
        for (int j = 0; j < sizey; j++){
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

                U_new[INDEX(i,j,k)] = Update_Calculus(U[INDEX(i,j,k)],FU[INDEX(i,j,k)],GU[INDEX(i,j,k)], 
                                                    U_up, U_down, U_left, U_right,
                                                    FU_up, FU_down, GU_left, GU_right,
                                                alpha_u, alpha_v, delta_t);

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

    int sizex = (int)(X / delta_x);
    int sizey = (int)(Y / delta_y);

    double *U = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *FU = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *GU = (double*)malloc(sizeof(double)*3*sizex*sizey);

    /* GOAL : SOLVING DIFFERENTIAL EQUATION 
            ∂U/∂t  + ∂F(U)/∂x + ∂G(U)/∂y = 0     
    */ 
    FILE *fp = fopen("shallow_water_simulation.json", "w");

    // store configuration 
    fprintf(fp, "{\n");
    fprintf(fp, "\"name\": \"lax friedrichs finite volume\",\n");
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
    double delta_t;
    double alpha_u , alpha_v;
    CNF_Condition(U , sizex, sizey, &delta_t , &alpha_u, &alpha_v);

    // initial step store 
    store_matrix(fp , U, step, sizex, sizey);

    while (count_time < time){
        double* U_new = (double*)malloc(sizeof(double)*3*sizex*sizey);

        Update_U(U_new, U, FU, GU, sizex, sizey, delta_t, alpha_u, alpha_v);

        // update U 
        for (int i = 0 ; i < sizex ;i++){
            for (int j = 0; j < sizey; j++){
                for (int k = 0; k < 3; k++){
                    U[INDEX(i,j,k)] = U_new[INDEX(i,j,k)];
                }
            }
        }
        
        // update flux F and Flux G 
        Update_FluxF(U, FU, sizex, sizey);
        Update_FluxG(U, GU, sizex ,sizey);

        step++;
        count_time += delta_t;

        // update CNF condition
        CNF_Condition(U,sizex, sizey, &delta_t, &alpha_u, &alpha_v);
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