#include <ctime>
#include <iostream>
#include <stdio.h>

const int TIMES = 1;
const int N = 8;
const int BLOCK_SIZE = 32;
const int AD_HOC = 8;

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c[tid] = a[tid] + b[tid];
}

__global__ void sub(int *a, int *b, int *c) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c[tid] = a[tid] - b[tid];
}

__global__ void sub_add(int *a1, int *b1, int *c1, int *a2, int *b2,
                        int *c2) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c1[tid] = a1[tid] - b1[tid];
  c2[tid] = a2[tid] + b2[tid];
}

__global__ void add_add(int *a1, int *b1, int *c1, int *a2, int *b2,
                        int *c2) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c1[tid] = a1[tid] + b1[tid];
  c2[tid] = a2[tid] + b2[tid];
}

__global__ void wadd(int *a, int *c1, int *c2) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c1[tid] += a[tid];
  c2[tid] += a[tid];
}

__global__ void wsub_add(int *a, int *c1, int *c2) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  c1[tid] -= a[tid];
  c2[tid] += a[tid];
}

__global__ void madd(int *a, int *b) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  b[tid] += a[tid];
}

__global__ void msub(int *a, int *b) {
  int tid = blockIdx.y * N * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE +
            threadIdx.y * N + threadIdx.x;
  b[tid] -= a[tid];
}

__global__ void cuda_multiplication(int *d_a, int *d_b, int *d_c,
                                    int size) {

  int i, j;
  int temp = 0;

  __shared__ int temp_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int temp_B[BLOCK_SIZE][BLOCK_SIZE];

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

      temp += temp_A[threadIdx.y][k] * temp_B[k][threadIdx.x];
    }
    // Synchronize
    __syncthreads();
  }

  d_c[row * size + col] = temp;
}


void cuda_strassen(int *a, int *b, int *c, int *t, int n) {

  if (n <= AD_HOC) {
    dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cuda_multiplication<<<dimGrid, dimBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();
  } else {
    int halfn = n / 2;
    dim3 grid(halfn / BLOCK_SIZE, halfn / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int p11 = 0, p12 = halfn, p21 = N * halfn, p22 = N * halfn + halfn;
    int *t1 = t;
    int *t2 = t + halfn * N;
    int *next = t + halfn;

    sub<<<grid, block>>>(&a[p21], &a[p11], &c[p12]);
    add<<<grid, block>>>(&b[p11], &b[p12], &c[p21]);
    cudaDeviceSynchronize();
    cuda_strassen(&c[p12], &c[p21], &c[p22], next, halfn);
    cudaDeviceSynchronize();

    sub<<<grid, block>>>(&a[p12], &a[p22], &c[p12]);
    add<<<grid, block>>>(&b[p21], &b[p22], &c[p21]);
    cudaDeviceSynchronize();
    cuda_strassen(&c[p12], &c[p21], &c[p11], next, halfn);
    cudaDeviceSynchronize();

    add<<<grid, block>>>(&a[p11], &a[p22], &c[p12]);
    add<<<grid, block>>>(&b[p11], &b[p22], &c[p21]);
    cudaDeviceSynchronize();
    cuda_strassen(&c[p12], &c[p21], t1, next, halfn);
    cudaDeviceSynchronize();

    wadd<<<grid, block>>>(t1, &c[p11], &c[p22]);
    add<<<grid, block>>>(&a[p21], &a[p22], t2);
    cudaDeviceSynchronize();
    cuda_strassen(&b[p11], &c[p21], t2, next, halfn);
    cudaDeviceSynchronize();

    msub<<<grid, block>>>(&c[p21], &c[p22]);
    sub<<<grid, block>>>(&b[p21], &b[p11], t1);
    cudaDeviceSynchronize();
    cuda_strassen(&a[p22], t1, t2, next, halfn);
    cudaDeviceSynchronize();

    wadd<<<grid, block>>>(t2, &c[p11], &c[p21]);
    sub<<<grid, block>>>(&b[p12], &b[p22], t1);
    cudaDeviceSynchronize();
    cuda_strassen(&a[p11], t1, &c[p12], next, halfn);
    cudaDeviceSynchronize();

    madd<<<grid, block>>>(&c[p12], &c[p22]);
    add<<<grid, block>>>(&a[p11], &a[p12], t2);
    cudaDeviceSynchronize();
    cuda_strassen(t2, &b[p22], t1, next, halfn);
    cudaDeviceSynchronize();

    wsub_add<<<grid, block>>>(t1, &c[p11], &c[p12]);
  }
}

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

int main(void) {
  srand(time(0));

  // Allocate memory on the host
  int *host_a, *host_b, *host_c;
  cudaMallocHost((void **)&host_a, sizeof(int) * N * N);
  cudaMallocHost((void **)&host_b, sizeof(int) * N * N);
  cudaMallocHost((void **)&host_c, sizeof(int) * N * N);


  int *device_a, *device_b, *device_c;
  // Allocate memory on the device
  cudaMalloc((void **)&device_a, sizeof(int) * N * N);
  cudaMalloc((void **)&device_b, sizeof(int) * N * N);
  cudaMalloc((void **)&device_c, sizeof(int) * N * N);

  // Fill matrix entries with random ints < 1
  generate_matrix(N, host_a);
  generate_matrix(N, host_b);

  // copy matrix a and b from host to device memory
  cudaMemcpy(device_a, host_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  clock_t st = clock();
  int *t;
  for (int i = 0; i < TIMES; i++) {
    cuda_strassen(device_a, device_b, device_c, t, N);
    // move result from device to host
    cudaMemcpy(host_c, device_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  // display_matrix(N, host_c);
  }
  float time_cuda = float(clock() - st) / (CLOCKS_PER_SEC * TIMES);
  // printf("%d %d %d", host_a[2], host_b[2], host_c[2]);

  display_matrix(N, host_c);
  std::cout << std::endl << "Matrix size: " << N << " * " << N;
  std::cout << std::endl << "Duration: " << time_cuda;
  std::cout << std::endl << "Block size: " << BLOCK_SIZE;
  std::cout << std::endl << "Ad_hoc size: " << AD_HOC;

  // Free memory on host and device
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_c);
  return 0;
}
