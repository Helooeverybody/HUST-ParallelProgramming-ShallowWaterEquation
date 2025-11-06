#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err_));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


// domain problem 
#define H  3.0
#define X  20.0
#define Y  20.0
#define g 9.81
#define delta_h 20.0 
#define initial_u 10.0
#define initial_v 10.0
#define delta_x 2.0
#define delta_y 2.0
#define CB_name 'R'
#define C 0.1

#define TIME_MAX 5.0 // Renamed from time to avoid conflict

#define INDEX(i, j, k) ((i) * ((int)(Y / delta_y)) * 3 + (j) * 3 + (k))

#define DEVICE_INDEX(i, j, k, current_sizey)                                   \
  ((i) * (current_sizey) * 3 + (j) * 3 + (k))

// Kernel launch configuration
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// __device__ function for the core calculation in Update_U
__device__ double Update_Calculus_device(
    double U_central, double FU_central, double GU_central, double U_up,
    double U_down, double U_left, double U_right, double FU_up, double FU_down,
    double GU_left, double GU_right, double alpha_u, double alpha_v,
    double delta_t_val /*renamed from delta_t to avoid conflict with define*/) {

  double Flux_margin_up = 0.5 * (FU_central - FU_up);
  double Flux_margin_down = 0.5 * (-FU_central + FU_down);
  double Flux_margin_left = 0.5 * (-GU_central + GU_left);
  double Flux_margin_right = 0.5 * (GU_central - GU_right);

  double Flux_u = (Flux_margin_down -
                   Flux_margin_up); // This seems to be related to F (x-flux)
  double Flux_v =
      (Flux_margin_left -
       Flux_margin_right); // This seems to be related to G (y-flux)
                           // Original comment mentioned u and v, but it's about
                           // F and G components typically


  double final_value = 0.25 * (U_down + U_left + U_up + U_right) -
                       (delta_t_val / delta_x) * Flux_u -
                       (delta_t_val / delta_y) * Flux_v;
  return final_value;
}

__global__ void Update_FluxF_kernel(double *U, double *FU, int sizex,
                                    int sizey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sizex && j < sizey) {
    double h_val = U[DEVICE_INDEX(i, j, 0, sizey)];
    // Avoid division by zero or very small h_val if U[DEVICE_INDEX(i, j, 0,
    // sizey)] is close to 0
    double u_val, v_val;
    if (h_val < 1e-6) { // Threshold for small h
      u_val = 0.0;
      v_val = 0.0;
    } else {
      v_val = U[DEVICE_INDEX(i, j, 1, sizey)] /
              h_val; // hu/h = u, hv/h = v. Original was U(1)=hv, U(2)=hu
      u_val = U[DEVICE_INDEX(i, j, 2, sizey)] /
              h_val; // So v from U(1), u from U(2)
    }

    FU[DEVICE_INDEX(i, j, 0, sizey)] = h_val * u_val;
    FU[DEVICE_INDEX(i, j, 1, sizey)] =
        h_val * u_val * u_val +
        0.5 * g * h_val * h_val; // This should be h*u*v for F(1) if U(1) is hv
                                 // Original C: U(0)=h, U(1)=hv, U(2)=hu
                                 // F(U) = [hu, hu^2 + 0.5gh^2, huv]^T
                                 // So FU(0) = hu (correct)
                                 // FU(1) = hu^2 + 0.5gh^2 (correct based on
                                 // u_val from U(2)) FU(2) = huv (correct)
    FU[DEVICE_INDEX(i, j, 2, sizey)] = h_val * u_val * v_val;
  }
}

__global__ void Update_FluxG_kernel(double *U, double *GU, int sizex,
                                    int sizey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sizex && j < sizey) {
    double h_val = U[DEVICE_INDEX(i, j, 0, sizey)];
    double u_val, v_val;

    if (h_val < 1e-6) {
      u_val = 0.0;
      v_val = 0.0;
    } else {
      v_val = U[DEVICE_INDEX(i, j, 1, sizey)] / h_val; // U(1) is hv
      u_val = U[DEVICE_INDEX(i, j, 2, sizey)] / h_val; // U(2) is hu
    }

    // G(U) = [hv, huv, hv^2 + 0.5gh^2]^T
    GU[DEVICE_INDEX(i, j, 0, sizey)] = h_val * v_val;
    GU[DEVICE_INDEX(i, j, 1, sizey)] = h_val * u_val * v_val;
    GU[DEVICE_INDEX(i, j, 2, sizey)] =
        h_val * v_val * v_val + 0.5 * g * h_val * h_val;
  }
}

__global__ void Calculate_Local_Speeds_kernel(double *U, double *local_u_speeds,
                                              double *local_v_speeds, int sizex,
                                              int sizey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sizex && j < sizey) {
    double h_val = U[DEVICE_INDEX(i, j, 0, sizey)];
    double u_val, v_val;
    if (h_val < 1e-6) { // Avoid issues with zero or negative h
      u_val = 0.0;
      v_val = 0.0;
      h_val = 0.0; // Ensure sqrt(g*h) is safe
    } else {
      v_val = U[DEVICE_INDEX(i, j, 1, sizey)] / h_val;
      u_val = U[DEVICE_INDEX(i, j, 2, sizey)] / h_val;
    }

    // Ensure h_val is non-negative for sqrt
    h_val = fmax(0.0, h_val);

    local_u_speeds[i * sizey + j] = fabs(u_val) + sqrt(g * h_val);
    local_v_speeds[i * sizey + j] = fabs(v_val) + sqrt(g * h_val);
  }
}

void CNF_Condition_cuda(double *d_U, int sizex, int sizey, double *delta_t_val,
                        double *alpha_u, double *alpha_v) {
  double *d_local_u_speeds, *d_local_v_speeds;
  double *h_local_u_speeds, *h_local_v_speeds;

  size_t array_size = sizex * sizey * sizeof(double);

  CUDA_CHECK(cudaMalloc((void **)&d_local_u_speeds, array_size));
  CUDA_CHECK(cudaMalloc((void **)&d_local_v_speeds, array_size));
  h_local_u_speeds = (double *)malloc(array_size);
  h_local_v_speeds = (double *)malloc(array_size);

  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
  dim3 gridDim((sizex + blockDim.x - 1) / blockDim.x,
               (sizey + blockDim.y - 1) / blockDim.y, 1);

  Calculate_Local_Speeds_kernel<<<gridDim, blockDim>>>(
      d_U, d_local_u_speeds, d_local_v_speeds, sizex, sizey);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_local_u_speeds, d_local_u_speeds, array_size,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_local_v_speeds, d_local_v_speeds, array_size,
                        cudaMemcpyDeviceToHost));

  double max_speed_u = 0.0;
  double max_speed_v = 0.0;

  for (int i = 0; i < sizex; i++) {
    for (int j = 0; j < sizey; j++) {
      if (h_local_u_speeds[i * sizey + j] > max_speed_u) {
        max_speed_u = h_local_u_speeds[i * sizey + j];
      }
      if (h_local_v_speeds[i * sizey + j] > max_speed_v) {
        max_speed_v = h_local_v_speeds[i * sizey + j];
      }
    }
  }

  *alpha_u = (max_speed_u < 1e-6)
                 ? 1.0
                 : max_speed_u; // Avoid division by zero if all speeds are tiny
  *alpha_v = (max_speed_v < 1e-6) ? 1.0 : max_speed_v;

  if (*alpha_u < 1e-6 ||
      *alpha_v <
          1e-6) { // If speeds are essentially zero, dt can be large but capped.
    *delta_t_val = fmin(delta_x, delta_y); // A reasonable large dt
  } else {
    *delta_t_val = C * fmin(delta_x / *alpha_u, delta_y / *alpha_v);
  }

  free(h_local_u_speeds);
  free(h_local_v_speeds);
  CUDA_CHECK(cudaFree(d_local_u_speeds));
  CUDA_CHECK(cudaFree(d_local_v_speeds));
}

__global__ void InitialConditionBound_kernel(double *U, int sizex, int sizey) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sizex && j < sizey) {
    double h_pert, v_pert, u_pert;
    if (i == 0) { // Original C code logic for perturbation
      h_pert = delta_h;
      v_pert = initial_v;
      u_pert = initial_u;
    } else {
      // Replicating original (double)(1/i) which is 0 for i > 1 due to int
      // division
      h_pert = ((double)(1 / i)) * delta_h;
      v_pert = ((double)(1 / i)) * initial_v;
      u_pert = ((double)(1 / i)) * initial_u;
    }
    U[DEVICE_INDEX(i, j, 0, sizey)] = (H + h_pert);
    U[DEVICE_INDEX(i, j, 1, sizey)] = (H + h_pert) * v_pert; // hv
    U[DEVICE_INDEX(i, j, 2, sizey)] = (H + h_pert) * u_pert; // hu
  }
}

__global__ void Update_U_kernel(double *U_new, double *U, double *FU,
                                double *GU, int sizex, int sizey,
                                double delta_t_val, double alpha_u,
                                double alpha_v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < sizex && j < sizey) {
    for (int k = 0; k < 3; k++) {
      double U_up, U_down, U_left, U_right;
      double FU_up, FU_down;
      double GU_left, GU_right;

      // Boundary conditions for U values
      // U_up (i+1)
      if (i == sizex - 1) { // Top boundary
        if (CB_name == 'R' && k == 2)
          U_up = -U[DEVICE_INDEX(i, j, k, sizey)]; // Reflect u component for hu
        else
          U_up = U[DEVICE_INDEX(
              i, j, k, sizey)]; // Neumann (du/dn = 0) or periodic for h, hv
      } else {
        U_up = U[DEVICE_INDEX(i + 1, j, k, sizey)];
      }

      // U_down (i-1)
      if (i == 0) { // Bottom boundary
        if (CB_name == 'R' && k == 2)
          U_down = -U[DEVICE_INDEX(i, j, k, sizey)];
        else
          U_down = U[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        U_down = U[DEVICE_INDEX(i - 1, j, k, sizey)];
      }

      // U_left (j-1)
      if (j == 0) { // Left boundary
        if (CB_name == 'R' && k == 1)
          U_left =
              -U[DEVICE_INDEX(i, j, k, sizey)]; // Reflect v component for hv
        else
          U_left = U[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        U_left = U[DEVICE_INDEX(i, j - 1, k, sizey)];
      }

      // U_right (j+1)
      if (j == sizey - 1) { // Right boundary
        if (CB_name == 'R' && k == 1)
          U_right = -U[DEVICE_INDEX(i, j, k, sizey)];
        else
          U_right = U[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        U_right = U[DEVICE_INDEX(i, j + 1, k, sizey)];
      }

      // Boundary conditions for FU values (used for F_i-1/2, F_i+1/2 terms)
      // FU_up (i+1)
      if (i == sizex - 1) {

        if (CB_name == 'R' && (k == 0 || k == 2))
          FU_up = -FU[DEVICE_INDEX(i, j, k, sizey)];
        else
          FU_up = FU[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        FU_up = FU[DEVICE_INDEX(i + 1, j, k, sizey)];
      }

      // FU_down (i-1)
      if (i == 0) {
        if (CB_name == 'R' && (k == 0 || k == 2))
          FU_down = -FU[DEVICE_INDEX(i, j, k, sizey)];
        else
          FU_down = FU[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        FU_down = FU[DEVICE_INDEX(i - 1, j, k, sizey)];
      }

      // Boundary conditions for GU values (used for G_j-1/2, G_j+1/2 terms)
      // GU_left (j-1)
      if (j == 0) {
        // G(U) = [hv, huv, hv^2 + 0.5gh^2]^T
        // k=0: hv. If v reflected (k=1 for U), then hv reflected.
        // k=1: huv. If v reflected, huv reflected.
        // k=2: hv^2 + 0.5gh^2. v^2 is not reflected.
        if (CB_name == 'R' && (k == 0 || k == 1))
          GU_left = -GU[DEVICE_INDEX(i, j, k, sizey)];
        else
          GU_left = GU[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        GU_left = GU[DEVICE_INDEX(i, j - 1, k, sizey)];
      }

      // GU_right (j+1)
      if (j == sizey - 1) {
        if (CB_name == 'R' && (k == 0 || k == 1))
          GU_right = -GU[DEVICE_INDEX(i, j, k, sizey)];
        else
          GU_right = GU[DEVICE_INDEX(i, j, k, sizey)];
      } else {
        GU_right = GU[DEVICE_INDEX(i, j + 1, k, sizey)];
      }

      U_new[DEVICE_INDEX(i, j, k, sizey)] = Update_Calculus_device(
          U[DEVICE_INDEX(i, j, k, sizey)], FU[DEVICE_INDEX(i, j, k, sizey)],
          GU[DEVICE_INDEX(i, j, k, sizey)], U_up, U_down, U_left, U_right,
          FU_up, FU_down, GU_left, GU_right, alpha_u, alpha_v, delta_t_val);
    }
  }
}

void store_matrix(FILE *fp, double *h_U, int step, int sizex, int sizey) {
  fprintf(fp, ",\"step_%d\": [\n", step);
  for (int i = 0; i < sizex; i++) {
    fprintf(fp, "  [");
    for (int j = 0; j < sizey; j++) {
      int idx = (i * sizey + j) * 3;
      double x = h_U[idx];
      double y = h_U[idx + 1];
      double z = h_U[idx + 2];
      fprintf(fp, "[%f, %f, %f]", x, y, z);
      if (j < sizey - 1)
        fprintf(fp, ", ");
    }
    fprintf(fp, "]");
    if (i < sizex - 1)
      fprintf(fp, ",\n");
    else
      fprintf(fp, "\n");
  }
  fprintf(fp, "]\n");
}

int main() {
  int sizex = (int)(X / delta_x);
  int sizey = (int)(Y / delta_y);
  size_t num_elements = 3 * sizex * sizey;
  size_t data_size = sizeof(double) * num_elements;

  double *h_U = (double *)malloc(data_size); // For storing results on host
  double *d_U_current, *d_U_next, *d_FU, *d_GU;

  CUDA_CHECK(cudaMalloc((void **)&d_U_current, data_size));
  CUDA_CHECK(cudaMalloc((void **)&d_U_next, data_size));
  CUDA_CHECK(cudaMalloc((void **)&d_FU, data_size));
  CUDA_CHECK(cudaMalloc((void **)&d_GU, data_size));

  FILE *fp = fopen("shallow_water_simulation.json", "w");
  if (!fp) {
    perror("Failed to open output file");
    return 1;
  }

  fprintf(fp, "{\n");
  fprintf(fp, "\"name\": \"lax friedrichs finite volume\",\n");
  fprintf(fp, "\"H\": %f,\n", H);
  fprintf(fp, "\"X\": %f,\n", X);
  fprintf(fp, "\"Y\": %f,\n", Y);
  fprintf(fp, "\"time\": %f,\n", TIME_MAX);
  fprintf(fp, "\"delta_x\": %f,\n", delta_x);
  fprintf(fp, "\"delta_y\": %f\n", delta_y);

  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
  dim3 gridDim((sizex + blockDim.x - 1) / blockDim.x,
               (sizey + blockDim.y - 1) / blockDim.y, 1);

  // Initial condition
  InitialConditionBound_kernel<<<gridDim, blockDim>>>(d_U_current, sizex,
                                                      sizey);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  Update_FluxF_kernel<<<gridDim, blockDim>>>(d_U_current, d_FU, sizex, sizey);
  CUDA_CHECK(cudaGetLastError());
  Update_FluxG_kernel<<<gridDim, blockDim>>>(d_U_current, d_GU, sizex, sizey);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int step = 0;
  double count_time = 0;
  double delta_t_val; // Renamed to avoid conflict
  double alpha_u, alpha_v;

  CNF_Condition_cuda(d_U_current, sizex, sizey, &delta_t_val, &alpha_u,
                     &alpha_v);

  CUDA_CHECK(cudaMemcpy(h_U, d_U_current, data_size, cudaMemcpyDeviceToHost));
  store_matrix(fp, h_U, step, sizex, sizey);
  printf("Step: %d, Time: %f, Delta_t: %f, alpha_u: %f, alpha_v: %f\n", step,
         count_time, delta_t_val, alpha_u, alpha_v);

  while (count_time < TIME_MAX) {
    Update_U_kernel<<<gridDim, blockDim>>>(d_U_next, d_U_current, d_FU, d_GU,
                                           sizex, sizey, delta_t_val, alpha_u,
                                           alpha_v);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Swap U_current and U_next pointers
    double *d_temp = d_U_current;
    d_U_current = d_U_next;
    d_U_next = d_temp;

    Update_FluxF_kernel<<<gridDim, blockDim>>>(d_U_current, d_FU, sizex, sizey);
    CUDA_CHECK(cudaGetLastError());
    Update_FluxG_kernel<<<gridDim, blockDim>>>(d_U_current, d_GU, sizex, sizey);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    step++;
    count_time += delta_t_val;

    CNF_Condition_cuda(d_U_current, sizex, sizey, &delta_t_val, &alpha_u,
                       &alpha_v);

    // Always store and print the current step's state as requested by user
    CUDA_CHECK(cudaMemcpy(h_U, d_U_current, data_size, cudaMemcpyDeviceToHost));
    store_matrix(fp, h_U, step, sizex, sizey);
    printf("Step: %d, Time: %f, Delta_t: %f, alpha_u: %f, alpha_v: %f\n", step,
           count_time, delta_t_val, alpha_u, alpha_v);

    if (delta_t_val <= 0) {
      printf("Error: delta_t is non-positive (%f). Exiting.\n", delta_t_val);
      break;
    }
    if (isnan(alpha_u) || isnan(alpha_v) || isinf(alpha_u) || isinf(alpha_v)) {
      printf("Error: alpha values are NaN or Inf. Exiting.\n");
      CUDA_CHECK(
          cudaMemcpy(h_U, d_U_current, data_size, cudaMemcpyDeviceToHost));
      store_matrix(fp, h_U, step, sizex, sizey); // Store last valid state
      break;
    }
  }

  fprintf(fp, ",\"total_step\": %d\n", step + 1);
  fprintf(fp, "}");
  fclose(fp);

  free(h_U);
  CUDA_CHECK(cudaFree(d_U_current));
  CUDA_CHECK(cudaFree(d_U_next));
  CUDA_CHECK(cudaFree(d_FU));
  CUDA_CHECK(cudaFree(d_GU));

  printf("CUDA simulation finished. Results saved to "
         "shallow_water_simulation_cuda.json\n");
  return 0;
}
