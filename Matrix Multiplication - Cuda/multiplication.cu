#include <ctime>
#include <iostream>
#include <stdio.h>
#include <vector>

const int TIMES = 10;
const int N = 2048;
const int BLOCK_SIZE = 32;

void generate_matrix(int size, int *matrix) {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      matrix[i * N + j] = rand() % 10;
    }
  }
}

void display_matrix(int size, int *matrix) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << matrix[i * size + j] << " ";
    }
    std::cout << std::endl;
  }
}

void old_multiplication(int size, int *m1, int *m2, int *m3) {
  int i, j, k;
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      for (k = 0; k < size; ++k) {
        m3[i * size + j] += m1[i * size + k] * m2[k * size + j];
      }
    }
  }
}

__global__ void cuda_multiplication(int *d_a, int *d_b, int *d_c, int size) {

  int i, j;
  float temp = 0;

  __shared__ float temp_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_B[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  for (int tile = 0; tile < gridDim.x; tile++) {

    // Row i of matrix a
    j = tile * BLOCK_SIZE + threadIdx.x;
    // Column j of matrix b
    i = tile * BLOCK_SIZE + threadIdx.y;

    // Load a[i][j] to shared memory
    temp_A[threadIdx.y][threadIdx.x] = d_a[row * size + j];

    // Load b[i][j] to shared memory
    temp_B[threadIdx.y][threadIdx.x] = d_b[i * size + col];

    // Synchronize before computation
    __syncthreads();

    // multiply one tile of res from tiles of a and b in shared memory
    for (int k = 0; k < BLOCK_SIZE; k++) {

      temp += temp_A[threadIdx.y][k] *
              temp_B[k][threadIdx.x]; 
    }
    // Synchronize
    __syncthreads();
  }

  d_c[row * size + col] = temp;
}

int main() {
  srand(time(0));

  // Allocate memory on the host
  int *host_a, *host_b, *host_c;
  cudaMallocHost((void **)&host_a, sizeof(int) * N * N);
  cudaMallocHost((void **)&host_b, sizeof(int) * N * N);
  cudaMallocHost((void **)&host_c, sizeof(int) * N * N);

  // Fill matrix entries with random digits
  generate_matrix(N, host_a);
  generate_matrix(N, host_b);

  // display_matrix(N, host_b);
  // display_matrix(N, host_a);

  int *device_a, *device_b, *device_c;
  // Allocate memory on the device
  cudaMalloc((void **)&device_a, sizeof(int) * N * N);
  cudaMalloc((void **)&device_b, sizeof(int) * N * N);
  cudaMalloc((void **)&device_c, sizeof(int) * N * N);

  // copy matrix a and b from host to device memory
  cudaMemcpy(device_a, host_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int gridevice_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(gridevice_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  clock_t st = clock();
  for (int i = 0; i < TIMES; i++) {
    cuda_multiplication<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
  // move result from device to host
    cudaMemcpy(host_c, device_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  }
  printf("%d %d", host_b[3], host_a[4]);

  printf("%d %d", host_c[3], host_c[4]);
  float time_cuda = float(clock() - st) / (CLOCKS_PER_SEC * TIMES);
  std::cout << std::endl << "Matrix size: " << N << " * " << N;
  std::cout << std::endl << "GPU: " << time_cuda;
  std::cout << std::endl << "Block size: " << BLOCK_SIZE;


  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  st = clock();
  old_multiplication(N, host_a, host_b, host_c);
  float time_cpu = float(clock() - st) / CLOCKS_PER_SEC;
  std::cout << std::endl << "CPU: " << time_cpu;
  std::cout << std::endl << "speedup: " << time_cpu / time_cuda;

  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_c);
  return 0;
}