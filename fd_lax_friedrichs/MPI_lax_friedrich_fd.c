// Following rows

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<mpi.h>

// domain problem 
#define H  10.0
#define X  8.0
#define Y  8.0
#define time 3.0
#define g 9.81
#define delta_h 15.0 
#define initial_u 10.0
#define initial_v 10.0
#define delta_x 0.5
#define delta_y 0.5
#define CB_name 'T'
#define C 0.5


#define INDEX(i, j, k) ((i)*((int)(Y/delta_y))*3 + (j)*3 + (k))


void store_matrix(FILE *fp , double* U, int step, int sizex , int sizey){
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

void Update_FluxF(double* U, double* FU,int sizex, int sizey){
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

void Update_FluxG(double* U, double* GU, int sizex, int sizey){
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
double Calculate_FluxF(double* U, int j, int k) {
    double h,u,v;
        h = U[j*3+0];  
        v = U[j*3+1] / U[j*3+0];  
        u = U[j*3+2] / U[j*3+0];  

    if (k == 0)
        return h*u;
    else if (k == 1){
        return h*u*u + 0.5f*g*h*h;}
    else
        return h*u*v;
}

double CNF_Condition(double* U, double sizex, double sizey){
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
    double delta_t = C*fmin(delta_x/max_speed_u, delta_y/max_speed_v);
    return delta_t;
}

void InitialConditionBound(double* U, double* FU, double* GU, int sizex , int sizey){
    // initial boundary condition 
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

    // update Flux F and Flux G matrix 
    Update_FluxF(U , FU, sizex , sizey);
    Update_FluxG(U, GU , sizex , sizey);
}



double Update_Calculus(double U_up , double U_down , double U_left, double U_right, double FU_up , double FU_down , double GU_left,double GU_right, double delta_t){

    double Flux_u = (FU_up - FU_down);
    double Flux_v = (GU_right - GU_left);
    double final_value = 0.25f*(U_up + U_down + U_left + U_right) - (delta_t/(2.0f*delta_x))*Flux_u - (delta_t/(2.0f*delta_y))*Flux_v;

    return final_value;
}
void Update_Border(double* U_border, double* U, int border, int sizey){
    for (int j = 0; j < sizey; j++)
        for(int k = 0; k < 3; k++){
            if (CB_name == 'R' && k == 2)
                U_border[3*j+k] = -U[INDEX(border,j,k)];
            else
                U_border[3*j+k] = U[INDEX(border,j,k)];
    }
}

void Update_U(double* U_new, double* U, double* FU , double* GU,double* Up, double* Ud, double delta_t, int sizexc, int sizey, int RANK, int NP){
    for (int i = 0; i < sizexc; i++){
        for (int j = 0; j < sizey; j++){
            for (int k = 0; k < 3; k++){
                double U_up, U_down , U_left, U_right;
                double FU_up , FU_down;
                double GU_left , GU_right;

                // boundary condition 

                // U_up 
                if (i == sizexc - 1) {
                   U_up = Up[3*j+k];
                } else {
                    U_up = U[INDEX(i+1,j,k)];
                }
                
                // U_down 
                if (i == 0) {
                    U_down = Ud[3*j+k];
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
                

                // FU
                if (RANK ==0){
                    // FU_up 
                    if (i == sizexc - 1) FU_up = Calculate_FluxF(Up,j,k);
                    else FU_up = FU[INDEX(i+1,j,k)];
                    // FU_down 
                    if (i == 0) {
                        if (CB_name == 'R' && k == 0) FU_down = -FU[INDEX(i,j,k)];
                        else if (CB_name == 'R' && k == 2) FU_down = -FU[INDEX(i,j,k)];
                        else FU_down = FU[INDEX(i,j,k)];
                    }else FU_down = FU[INDEX(i-1,j,k)];
                    
                }else if (RANK == NP-1){
                     // FU_up 
                    if (i == sizexc - 1) {
                        if (CB_name == 'R' && k == 0) FU_up = -FU[INDEX(i,j,k)];
                        else if (CB_name == 'R' && k == 2) FU_up = -FU[INDEX(i,j,k)];
                        else FU_up = FU[INDEX(i,j,k)];
                    } else FU_up = FU[INDEX(i+1,j,k)];
                    // FU_down 
                    if (i == 0) FU_down = Calculate_FluxF(Ud,j,k);
                    else FU_down = FU[INDEX(i-1,j,k)];   
                }else{
                    if (i == 0) 
                        {
                            FU_down = Calculate_FluxF(Ud,j,k);
                            FU_up = FU[INDEX(i+1,j,k)];   
                        }
                    else if (i == sizexc - 1) {
                            FU_up = Calculate_FluxF(Up,j,k);
                            FU_down = FU[INDEX(i-1,j,k)]; 
                    }
                    else{
                        FU_up = FU[INDEX(i+1,j,k)];
                        FU_down = FU[INDEX(i-1,j,k)];  
                    }
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
                U_new[INDEX(i,j,k)] = Update_Calculus(U_up , U_down, U_left, U_right, FU_up , FU_down, GU_left, GU_right, delta_t);    
                
            } 
        }
    }
}


int main(int argc, char *argv[]){
    int NP, RANK;  
    MPI_Init(&argc, &argv);
    MPI_Status state;
    MPI_Comm_size(MPI_COMM_WORLD, &NP);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
   
    int sizex = (int)(X / delta_x);
    int sizey = (int)(Y / delta_y);

    double *U  = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *FU = (double*)malloc(sizeof(double)*3*sizex*sizey);
    double *GU = (double*)malloc(sizeof(double)*3*sizex*sizey);

    int step = 0;
    double count_time = 0;
    double delta_t, min_delta;
    FILE* fp;
    InitialConditionBound(U,FU,GU, sizex, sizey);
    delta_t = CNF_Condition(U , sizex, sizey);
  
    if(RANK == 0){
        //FILE *fp = fopen("D:/MINH/Project/shallow_water_equation/shallow_water_simulation.json", "w");
        fp = fopen("shallow_water_simulation.json", "w");
        if (fp == NULL) 
            perror("Failed to open file!!!");
        // store configuration 
        fprintf(fp, "{\n");
        fprintf(fp, "\"H\": %f,\n", H);
        fprintf(fp, "\"X\": %f,\n", X);
        fprintf(fp, "\"Y\": %f,\n", Y);
        fprintf(fp, "\"time\": %f,\n", time);
        fprintf(fp, "\"delta_x\": %f,\n", delta_x);
        fprintf(fp, "\"delta_y\": %f\n", delta_y);
        // initial step store 
        store_matrix(fp, U, step, sizex, sizey);
    }
    // initial state


    // domain decomposition
    int sizexc;
    double *Uc, *FUc, *GUc;
    sizexc =  (int)(sizex/NP);
    Uc  = (double*)malloc(sizeof(double)*3*sizexc*sizey);
    FUc = (double*)malloc(sizeof(double)*3*sizexc*sizey);
    GUc = (double*)malloc(sizeof(double)*3*sizexc*sizey);

    double* Up, *Ud;
    Up = (double*)malloc(sizeof(double)*3*sizey);
    Ud = (double*)malloc(sizeof(double)*3*sizey);
      
    MPI_Scatter(U, 3*sizexc*sizey, MPI_DOUBLE, Uc, 3*sizexc*sizey, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(FU,3*sizexc*sizey, MPI_DOUBLE, FUc,3*sizexc*sizey, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(GU,3*sizexc*sizey, MPI_DOUBLE, GUc,3*sizexc*sizey, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    while (count_time < time){
        
        double *deltas = (RANK == 0)? malloc(sizeof(double) * NP): NULL;
        // communicate from down to up
        if (RANK == 0){
            Update_Border(Ud, Uc, 0, sizey);
            MPI_Send(Uc+3*sizey*(sizexc-1), sizey*3, MPI_DOUBLE,RANK+1, RANK, MPI_COMM_WORLD);
        }else if(RANK == NP-1){
            MPI_Recv(Ud, sizey*3, MPI_DOUBLE, RANK-1, RANK-1, MPI_COMM_WORLD,&state);
        }else{
            MPI_Send(Uc+3*sizey*(sizexc-1), sizey*3, MPI_DOUBLE,RANK+1, RANK, MPI_COMM_WORLD);
            MPI_Recv(Ud, sizey*3, MPI_DOUBLE, RANK-1, RANK-1, MPI_COMM_WORLD,&state);
        }
      
        //communicate from up to down

        if (RANK == 0){
            MPI_Recv(Up, sizey*3, MPI_DOUBLE, RANK+1, RANK+1, MPI_COMM_WORLD,&state);
        }else if(RANK == NP-1){
            Update_Border(Up, Uc, sizexc-1, sizey);
            MPI_Send(Uc, sizey*3, MPI_DOUBLE,RANK-1, RANK, MPI_COMM_WORLD);
        }else{
            MPI_Send(Uc, sizey*3, MPI_DOUBLE, RANK-1, RANK, MPI_COMM_WORLD);
            MPI_Recv(Up, sizey*3, MPI_DOUBLE, RANK+1, RANK+1, MPI_COMM_WORLD,&state);
        }
       
        double* U_new = (double*)malloc(sizeof(double)*3*sizexc*sizey);
        Update_U(U_new, Uc, FUc, GUc, Up, Ud, delta_t, sizexc, sizey, RANK, NP);
        for (int i = 0 ; i < sizexc ;i++)
            for (int j = 0; j < sizey; j++)
                for (int k = 0; k < 3; k++)
                    Uc[INDEX(i,j,k)] = U_new[INDEX(i,j,k)];
                
            
        
         // update flux F
        Update_FluxF(Uc, FUc, sizexc, sizey);
        // update flux G
        Update_FluxG(Uc, GUc, sizexc ,sizey);
        step++;

        // update delta_t 
        delta_t = CNF_Condition(Uc,sizexc, sizey);

        MPI_Gather(&delta_t, 1, MPI_DOUBLE, deltas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (RANK == 0) {
            min_delta = deltas[0];
            for (int p = 1; p < NP; ++p)
                if (deltas[p] < min_delta)
                    min_delta = deltas[p];
            delta_t = min_delta;
            for (int p = 1; p < NP; ++p)
                MPI_Send(&delta_t, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        } else 
            MPI_Recv(&delta_t, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        count_time += delta_t;
         // gather 
        MPI_Gather(Uc, 3*sizexc*sizey, MPI_DOUBLE, U, 3*sizexc*sizey, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       

        // store matrix state 
        if(RANK ==0)
            store_matrix(fp,U, step, sizex, sizey);
        
        free(U_new);
        free(deltas);
    }
    fprintf(fp,",\"total_step\": %d \n", step+1);
    // final store 
    fprintf(fp,"}");

    free(U);
    free(FU);
    free(GU);


    return 0;
    
}
