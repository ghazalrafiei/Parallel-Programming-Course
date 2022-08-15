#include <ctime>
#include <iostream>
#include <omp.h>
#include <vector>

const int TIMES = 50;
const int NUM_THREADS = 16;

typedef std::vector<std::vector<int>> Matrix;

void generate_matrix(int size, Matrix &matrix) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix[i][j] = rand() % 10;
    }
  }
}

void display(int size, Matrix &matrix) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

void multiplication(int size, Matrix &m1, Matrix &m2, Matrix &m3) {
  int i, j, k;
#pragma omp parallel for private(i, k, j) schedule(runtime)                    \
    num_threads(NUM_THREADS)
  // schedule options: static, runtime, auto, static, dynamic, guided
  for (i = 0; i < size; ++i) {
    for (j = 0; j < size; ++j) {
      for (k = 0; k < size; ++k) {
        m3[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
}

int main() {

  srand(time(0));

  int size = 256;
  // std::cin >> size;

  std::vector<int> in_vec(size, 0);
  Matrix m1(size, in_vec), m2(size, in_vec), m3(size, in_vec);

  generate_matrix(size, m1);
  generate_matrix(size, m2);

  float total_time = 0;

  for (int i = 0; i < TIMES; i++) {
#ifdef _OPENMP
    float t1 = omp_get_wtime();

#else
    clock_t st = clock();
#endif

    multiplication(size, m1, m2, m3);

#ifdef _OPENMP
    float t2 = omp_get_wtime();
    total_time += t2 - t1;

#else
    clock_t ed = clock();
    total_time += float(ed - st) / CLOCKS_PER_SEC;
#endif
  }

  std::cout << "mode: ";
#ifdef _OPENMP
  std::cout << "parallel" << std::endl;
#else
  std::cout << "sequential" << std::endl;
#endif
  std::cout << "average time: " << total_time / TIMES << std::endl;
  std::cout << "threads: " << NUM_THREADS << std::endl;
  std::cout << "matrix size: " << size << " * " << size << std::endl;
  std::cout << "times of run: " << TIMES << std::endl;
  return 0;
}