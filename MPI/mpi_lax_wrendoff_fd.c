#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>


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

#define LOCAL_INDEX(i, j, k, local_gsy) ((i) * (local_gsy) * 3 + (j) * 3 + (k))
#define GLOBAL_INDEX(i, j, k, global_gsy) ((i) * (global_gsy) * 3 + (j) * 3 + (k))


void Update_FluxF(double *U_local, double *FU_local, int sizexc, int sizey) {
    for (int i = 0; i < sizexc; i++) {
        for (int j = 0; j < sizey; j++) {
            if (U_local[LOCAL_INDEX(i, j, 0, sizey)] < 1e-9) {
                 FU_local[LOCAL_INDEX(i, j, 0, sizey)] = 0.0;
                 FU_local[LOCAL_INDEX(i, j, 1, sizey)] = 0.0;
                 FU_local[LOCAL_INDEX(i, j, 2, sizey)] = 0.0;
                 continue;
            }
            double h = U_local[LOCAL_INDEX(i, j, 0, sizey)];
            double v = U_local[LOCAL_INDEX(i, j, 1, sizey)] / h; // hv is U[1]
            double u = U_local[LOCAL_INDEX(i, j, 2, sizey)] / h; // hu is U[2]

            FU_local[LOCAL_INDEX(i, j, 0, sizey)] = h * u;
            FU_local[LOCAL_INDEX(i, j, 1, sizey)] = h * u * u + 0.5f * g * h * h;
            FU_local[LOCAL_INDEX(i, j, 2, sizey)] = h * u * v;
        }
    }
}

void Update_FluxG(double *U_local, double *GU_local, int sizexc, int sizey) {
    for (int i = 0; i < sizexc; i++) {
        for (int j = 0; j < sizey; j++) {
            if (U_local[LOCAL_INDEX(i, j, 0, sizey)] < 1e-9) {
                 GU_local[LOCAL_INDEX(i, j, 0, sizey)] = 0.0;
                 GU_local[LOCAL_INDEX(i, j, 1, sizey)] = 0.0;
                 GU_local[LOCAL_INDEX(i, j, 2, sizey)] = 0.0;
                 continue;
            }
            double h = U_local[LOCAL_INDEX(i, j, 0, sizey)];
            double v = U_local[LOCAL_INDEX(i, j, 1, sizey)] / h; // hv is U[1]
            double u = U_local[LOCAL_INDEX(i, j, 2, sizey)] / h; // hu is U[2]

            GU_local[LOCAL_INDEX(i, j, 0, sizey)] = h * v;
            GU_local[LOCAL_INDEX(i, j, 1, sizey)] = h * u * v;
            GU_local[LOCAL_INDEX(i, j, 2, sizey)] = h * v * v + 0.5 * g * h * h;
        }
    }
}

double CNF_Condition(double *U_local, int sizexc, int sizey, double *local_alpha_u_ptr, double *local_alpha_v_ptr) {
    double max_speed_u_local = 1e-9;
    double max_speed_v_local = 1e-9;

    for (int i = 0; i < sizexc; i++) {
        for (int j = 0; j < sizey; j++) {
            if (U_local[LOCAL_INDEX(i, j, 0, sizey)] < 1e-9) continue;
            double h = U_local[LOCAL_INDEX(i, j, 0, sizey)];
            double v = U_local[LOCAL_INDEX(i, j, 1, sizey)] / h;
            double u = U_local[LOCAL_INDEX(i, j, 2, sizey)] / h;

            double current_speed_v = fabs(v) + sqrt(g * h);
            if (current_speed_v > max_speed_v_local) max_speed_v_local = current_speed_v;
            double current_speed_u = fabs(u) + sqrt(g * h);
            if (current_speed_u > max_speed_u_local) max_speed_u_local = current_speed_u;
        }
    }
    *local_alpha_u_ptr = max_speed_u_local;
    *local_alpha_v_ptr = max_speed_v_local;
    if (max_speed_u_local < 1e-9) max_speed_u_local = 1e-9;
    if (max_speed_v_local < 1e-9) max_speed_v_local = 1e-9;
    return C * fmin(delta_x / max_speed_u_local, delta_y / max_speed_v_local);
}

void Get_Boundary_State_from_Local(double *U_border_target_row, double *U_source_local, int border_idx_local, int sizey) {
    for (int j = 0; j < sizey; j++) {
        for (int k_comp = 0; k_comp < 3; k_comp++) {
            if (CB_name == 'R' && k_comp == 2) {
                U_border_target_row[3 * j + k_comp] = -U_source_local[LOCAL_INDEX(border_idx_local, j, k_comp, sizey)];
            } else {
                U_border_target_row[3 * j + k_comp] =  U_source_local[LOCAL_INDEX(border_idx_local, j, k_comp, sizey)];
            }
        }
    }
}

double Calculate_FluxF_component_from_flat_state_row(double *U_flat_state_row, int j_in_row, int k_flux_comp) {
    double h, u, v;
    if (U_flat_state_row[j_in_row * 3 + 0] < 1e-9) return 0.0;
    h = U_flat_state_row[j_in_row * 3 + 0];
    v = U_flat_state_row[j_in_row * 3 + 1] / h;
    u = U_flat_state_row[j_in_row * 3 + 2] / h;

    if (k_flux_comp == 0) return h * u;
    else if (k_flux_comp == 1) return h * u * u + 0.5f * g * h * h;
    else if (k_flux_comp == 2) return h * u * v;
    return 0.0;
}
double Calculate_FluxG_component_from_flat_state_row(double *U_flat_state_row, int j_in_row, int k_flux_comp) {
    double h, u, v;
    if (U_flat_state_row[j_in_row * 3 + 0] < 1e-9) return 0.0;
    h = U_flat_state_row[j_in_row * 3 + 0];
    v = U_flat_state_row[j_in_row * 3 + 1] / h;
    u = U_flat_state_row[j_in_row * 3 + 2] / h;

    if (k_flux_comp == 0) return h * v;
    else if (k_flux_comp == 1) return h * u * v;
    else if (k_flux_comp == 2) return h * v * v + 0.5 * g * h * h;
    return 0.0;
}


void InitialConditionBound(double *U_local, double *FU_local, double *GU_local,
                           int sizexc, int sizey, int RANK, int NP, int start_ix_global) {
    for (int i = 0; i < sizexc; i++) {
        for (int j = 0; j < sizey; j++) {
            double h_pert, v_val, u_val;
            int current_global_i = start_ix_global + i;

            if (current_global_i == 0) {
                h_pert = delta_h;
                v_val = initial_v;
                u_val = initial_u;
            } else { 
                double factor_from_inv_i = (current_global_i == 1) ? 1.0 : 0.0;

                h_pert = factor_from_inv_i * delta_h;
                v_val = factor_from_inv_i * initial_v;
                u_val = factor_from_inv_i * initial_u;
            }
            U_local[LOCAL_INDEX(i, j, 0, sizey)] = H + h_pert;
            U_local[LOCAL_INDEX(i, j, 1, sizey)] = (H + h_pert) * v_val;
            U_local[LOCAL_INDEX(i, j, 2, sizey)] = (H + h_pert) * u_val;
        }
    }
    Update_FluxF(U_local, FU_local, sizexc, sizey);
    Update_FluxG(U_local, GU_local, sizexc, sizey);
}

void Update_half_step_FluxF(double *U_half, double *FU_half) {
    if (U_half[0] < 1e-9) { FU_half[0]=0; FU_half[1]=0; FU_half[2]=0; return; }
    double h = U_half[0]; 
    double v = U_half[1] / h;
    double u = U_half[2] / h;
    FU_half[0] = h * u;
    FU_half[1] = h * u * u + 0.5f * g * h * h;
    FU_half[2] = h * u * v;
}
void Update_half_step_FluxG(double *U_half, double *GU_half) {
    if (U_half[0] < 1e-9) { GU_half[0]=0; GU_half[1]=0; GU_half[2]=0; return; }
    double h = U_half[0];
    double v = U_half[1] / h;
    double u = U_half[2] / h;
    GU_half[0] = h * v;
    GU_half[1] = h * u * v;
    GU_half[2] = h * v * v + 0.5 * g * h * h;
}

void Update_U(double *U_new_local, double *U_local, double *FU_local, double *GU_local,
              double *U_halo_up_state_row, double *U_halo_down_state_row, double delta_t_global,
              int sizexc, int sizey, double global_alpha_x, double global_alpha_y,
              int RANK, int NP) {
    for (int i = 0; i < sizexc; i++) {
        for (int j = 0; j < sizey; j++) {
            double U_half_up[3], U_half_down[3], U_half_left[3], U_half_right[3];
            double U_entropy_x[3], U_entropy_y[3];

            for (int k_comp = 0; k_comp < 3; k_comp++) {
                double U_up_neighbor_k, U_down_neighbor_k, U_left_neighbor_k, U_right_neighbor_k;
                double FU_up_interface_k, FU_down_interface_k;
                double GU_left_interface_k, GU_right_interface_k;

                if (i == sizexc - 1) {
                    U_up_neighbor_k = U_halo_up_state_row[3 * j + k_comp];
                } else {
                    U_up_neighbor_k = U_local[LOCAL_INDEX(i + 1, j, k_comp, sizey)];
                }
                if (i == 0) {
                    U_down_neighbor_k = U_halo_down_state_row[3 * j + k_comp];
                } else {
                    U_down_neighbor_k = U_local[LOCAL_INDEX(i - 1, j, k_comp, sizey)];
                }
                if (j == 0) {
                    if (CB_name == 'R' && k_comp == 1) U_left_neighbor_k = -U_local[LOCAL_INDEX(i, j, k_comp, sizey)];
                    else U_left_neighbor_k = U_local[LOCAL_INDEX(i, j, k_comp, sizey)];
                } else {
                    U_left_neighbor_k = U_local[LOCAL_INDEX(i, j - 1, k_comp, sizey)];
                }
                if (j == sizey - 1) {
                    if (CB_name == 'R' && k_comp == 1) U_right_neighbor_k = -U_local[LOCAL_INDEX(i, j, k_comp, sizey)];
                    else U_right_neighbor_k = U_local[LOCAL_INDEX(i, j, k_comp, sizey)];
                } else {
                    U_right_neighbor_k = U_local[LOCAL_INDEX(i, j + 1, k_comp, sizey)];
                }

                double FU_current_k = FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                double GU_current_k = GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];

                if (i == 0) {
                    if (RANK == 0) {
                        FU_down_interface_k = FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        if (CB_name == 'R') {
                            if (k_comp == 0) FU_down_interface_k = -FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                            else if (k_comp == 2) FU_down_interface_k = -FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        }
                    } else {
                        FU_down_interface_k = Calculate_FluxF_component_from_flat_state_row(U_halo_down_state_row, j, k_comp);
                    }
                } else {
                    FU_down_interface_k = FU_local[LOCAL_INDEX(i-1,j,k_comp,sizey)];
                }

                if (i == sizexc - 1) {
                    if (RANK == NP - 1) {
                        FU_up_interface_k = FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        if (CB_name == 'R') {
                            if (k_comp == 0) FU_up_interface_k = -FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                            else if (k_comp == 2) FU_up_interface_k = -FU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        }
                    } else {
                        FU_up_interface_k = Calculate_FluxF_component_from_flat_state_row(U_halo_up_state_row, j, k_comp);
                    }
                } else {
                    FU_up_interface_k = FU_local[LOCAL_INDEX(i+1,j,k_comp,sizey)];
                }

                if (j == 0) {
                    GU_left_interface_k = GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                    if (CB_name == 'R') {
                        if (k_comp == 0) GU_left_interface_k = -GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        else if (k_comp == 1) GU_left_interface_k = -GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                    }
                } else {
                    GU_left_interface_k = GU_local[LOCAL_INDEX(i,j-1,k_comp,sizey)];
                }

                if (j == sizey - 1) {
                    GU_right_interface_k = GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                     if (CB_name == 'R') {
                        if (k_comp == 0) GU_right_interface_k = -GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                        else if (k_comp == 1) GU_right_interface_k = -GU_local[LOCAL_INDEX(i,j,k_comp,sizey)];
                    }
                } else {
                    GU_right_interface_k = GU_local[LOCAL_INDEX(i,j+1,k_comp,sizey)];
                }
                
                U_half_up[k_comp]   = 0.5 * (U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_up_neighbor_k)   - (delta_t_global / (2.0 * delta_x)) * (FU_up_interface_k - FU_current_k);
                U_half_down[k_comp] = 0.5 * (U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_down_neighbor_k) - (delta_t_global / (2.0 * delta_x)) * (FU_down_interface_k - FU_current_k);
                U_half_left[k_comp] = 0.5 * (U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_left_neighbor_k) - (delta_t_global / (2.0 * delta_y)) * (GU_left_interface_k - GU_current_k);
                U_half_right[k_comp]= 0.5 * (U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_right_neighbor_k)- (delta_t_global / (2.0 * delta_y)) * (GU_right_interface_k - GU_current_k);

                U_entropy_x[k_comp] = (U_up_neighbor_k - 2 * U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_down_neighbor_k);
                U_entropy_y[k_comp] = (U_right_neighbor_k - 2 * U_local[LOCAL_INDEX(i, j, k_comp, sizey)] + U_left_neighbor_k);
            }

            double FU_half_up[3], FU_half_down[3], GU_half_left[3], GU_half_right[3];
            Update_half_step_FluxF(U_half_up, FU_half_up);
            Update_half_step_FluxF(U_half_down, FU_half_down);
            Update_half_step_FluxG(U_half_left, GU_half_left);
            Update_half_step_FluxG(U_half_right, GU_half_right);

            for (int k_comp = 0; k_comp < 3; k_comp++) {
                U_new_local[LOCAL_INDEX(i, j, k_comp, sizey)] = U_local[LOCAL_INDEX(i, j, k_comp, sizey)]
                                           - (delta_t_global / delta_x) * (FU_half_up[k_comp] - FU_half_down[k_comp])
                                           - (delta_t_global / delta_y) * (GU_half_right[k_comp] - GU_half_left[k_comp])
                                           + global_alpha_x * 0.01 * U_entropy_x[k_comp]
                                           + global_alpha_y * 0.01 * U_entropy_y[k_comp];
            }
        }
    }
}

void store_matrix(FILE *fp, double *U_global_arr, int step, int g_sizex, int g_sizey) {
    fprintf(fp, ",\"step_%d\": [\n", step);
    for (int i = 0; i < g_sizex; i++) {
        fprintf(fp, "  [");
        for (int j = 0; j < g_sizey; j++) {
            double h_val = U_global_arr[GLOBAL_INDEX(i, j, 0, g_sizey)];
            double hv_val = U_global_arr[GLOBAL_INDEX(i, j, 1, g_sizey)];
            double hu_val = U_global_arr[GLOBAL_INDEX(i, j, 2, g_sizey)];
            fprintf(fp, "[%f, %f, %f]", h_val, hv_val, hu_val);
            if (j < g_sizey - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "]");
        if (i < g_sizex - 1) fprintf(fp, ",\n");
        else fprintf(fp, "\n");
    }
    fprintf(fp, "]\n");
}

int main(int argc, char *argv[]) {
    int RANK, NP;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NP);

    if (NP < 1 ) {
        if (RANK == 0) printf("This program requires at least 1 process\n");
        MPI_Finalize(); return 1;
    }

    int sizex = (int)(X / delta_x);
    int sizey = (int)(Y / delta_y);

    int base_sizexc = sizex / NP;
    int remainder_cells = sizex % NP;
    int sizexc = base_sizexc + (RANK < remainder_cells ? 1 : 0);
    int start_ix_global;
    if (RANK < remainder_cells) {
        start_ix_global = RANK * (base_sizexc + 1);
    } else {
        start_ix_global = remainder_cells * (base_sizexc + 1) + (RANK - remainder_cells) * base_sizexc;
    }

    double *U_global_arr = NULL;
    if (RANK == 0) {
        U_global_arr = (double *)malloc(sizeof(double) * 3 * sizex * sizey);
        if (!U_global_arr) { MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    double *Uc_local = (double *)malloc(sizeof(double) * 3 * sizexc * sizey);
    double *FUc_local = (double *)malloc(sizeof(double) * 3 * sizexc * sizey);
    double *GUc_local = (double *)malloc(sizeof(double) * 3 * sizexc * sizey);
    double *U_halo_up_state_row = (double *)malloc(sizeof(double) * 3 * sizey);
    double *U_halo_down_state_row = (double *)malloc(sizeof(double) * 3 * sizey);

    if (!Uc_local || !FUc_local || !GUc_local || !U_halo_up_state_row || !U_halo_down_state_row) {
        printf("Rank %d: Malloc failed for local/halo arrays.\n", RANK);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE *fp = NULL;
    if (RANK == 0) {
        fp = fopen("shallow_water_simulation.json", "w");
        if (!fp) { perror("Failed to open file"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fprintf(fp, "{\n\"H\": %f,\n\"X\": %f,\n\"Y\": %f,\n\"time\": %f,\n\"delta_x\": %f,\n\"delta_y\": %f,\n",
                H, X, Y, time, delta_x, delta_y);
        fprintf(fp, "\"name\": \"lax wendroff finite difference\" \n");
    }

    InitialConditionBound(Uc_local, FUc_local, GUc_local, sizexc, sizey, RANK, NP, start_ix_global);

    int *recvcounts = NULL; int *displs = NULL;
    if (RANK == 0) {
        recvcounts = (int *)malloc(NP * sizeof(int));
        displs = (int *)malloc(NP * sizeof(int));
        if(!recvcounts || !displs) { printf("Malloc failed for recvcounts/displs.\n"); MPI_Abort(MPI_COMM_WORLD,1);}
        int current_displ = 0;
        for (int p = 0; p < NP; p++) {
            int p_sizexc = sizex / NP + (p < sizex % NP ? 1 : 0);
            recvcounts[p] = 3 * p_sizexc * sizey;
            displs[p] = current_displ;
            current_displ += recvcounts[p];
        }
    }
    MPI_Gatherv(Uc_local, 3 * sizexc * sizey, MPI_DOUBLE, U_global_arr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (RANK == 0) store_matrix(fp, U_global_arr, 0, sizex, sizey);

    int step = 0;
    double count_time = 0;
    double delta_t_global;
    double local_alpha_u, local_alpha_v, global_alpha_x, global_alpha_y;
    MPI_Status status;

    while (count_time < time) {
        if (NP > 1) {
            if (RANK < NP - 1) MPI_Send(Uc_local + 3 * sizey * (sizexc - 1), 3 * sizey, MPI_DOUBLE, RANK + 1, 0, MPI_COMM_WORLD);
            if (RANK > 0)    MPI_Recv(U_halo_down_state_row, 3 * sizey, MPI_DOUBLE, RANK - 1, 0, MPI_COMM_WORLD, &status);
            if (RANK > 0)    MPI_Send(Uc_local, 3 * sizey, MPI_DOUBLE, RANK - 1, 1, MPI_COMM_WORLD);
            if (RANK < NP - 1) MPI_Recv(U_halo_up_state_row, 3 * sizey, MPI_DOUBLE, RANK + 1, 1, MPI_COMM_WORLD, &status);
        }

        if (RANK == 0) Get_Boundary_State_from_Local(U_halo_down_state_row, Uc_local, 0, sizey);
        if (RANK == NP - 1) Get_Boundary_State_from_Local(U_halo_up_state_row, Uc_local, sizexc - 1, sizey);

        double *U_new_local = (double *)malloc(sizeof(double) * 3 * sizexc * sizey);
        if (!U_new_local) { printf("Rank %d: Malloc failed for U_new_local.\n", RANK); MPI_Abort(MPI_COMM_WORLD, 1); }
        
        double local_delta_t = CNF_Condition(Uc_local, sizexc, sizey, &local_alpha_u, &local_alpha_v);
        MPI_Allreduce(&local_delta_t, &delta_t_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_alpha_u, &global_alpha_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_alpha_v, &global_alpha_y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        Update_U(U_new_local, Uc_local, FUc_local, GUc_local, U_halo_up_state_row, U_halo_down_state_row,
                 delta_t_global, sizexc, sizey, global_alpha_x, global_alpha_y, RANK, NP);

        memcpy(Uc_local, U_new_local, sizeof(double) * 3 * sizexc * sizey);
        free(U_new_local);

        Update_FluxF(Uc_local, FUc_local, sizexc, sizey);
        Update_FluxG(Uc_local, GUc_local, sizexc, sizey);
        
        step++;
        count_time += delta_t_global;

        MPI_Gatherv(Uc_local, 3 * sizexc * sizey, MPI_DOUBLE, U_global_arr, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (RANK == 0) store_matrix(fp, U_global_arr, step, sizex, sizey);

        if (count_time >= time) break; 
    }

    if (RANK == 0) {
        fprintf(fp, ",\"total_step\": %d\n", step +1 );
        fprintf(fp, "}");
        fclose(fp);
        free(U_global_arr);
        if (recvcounts) free(recvcounts);
        if (displs) free(displs);
    }

    free(Uc_local); free(FUc_local); free(GUc_local);
    free(U_halo_up_state_row); free(U_halo_down_state_row);
    MPI_Finalize();
    return 0;
}