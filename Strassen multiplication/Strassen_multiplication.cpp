// Got help from
// https://github.com/alielabridi/Strassen-Algorithm-parallelization-charmplusplus-/blob/master/OpenMP-Strassen.cpp

#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

const int SIZE = 1024;
const int AD_HOC = 1024;
const int NUM_THREADS = 16;
const int TIMES = 10;

typedef std::vector<std::vector<int>> Matrix;

void read_matrix(Matrix &A, std::string fname) {
  std::ifstream file;
  file.open(fname);
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      char x;
      file >> x;
      if (x - '0' > 0) {
        A[i][j] = x - '0';
      }
    }
  }
}

// Parallel Addition
void add(Matrix &mat1, Matrix &mat2, Matrix &ans, int size) {
  int i = 0, j = 0;
#pragma omp parallel for private(i, j)
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      ans[i][j] = mat1[i][j] + mat2[i][j];
    }
  }
}

// Parallel Subtraction
void subtract(Matrix &mat1, Matrix &mat2, Matrix &ans, int size) {
  int i = 0, j = 0;

#pragma omp parallel for private(i, j)

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      ans[i][j] = mat1[i][j] - mat2[i][j];
    }
  }
}

// Parallel Multiplication
void multiply(Matrix &mat1, Matrix &mat2, Matrix &ans, int size) {
  int i = 0, j = 0, k = 0;

#pragma omp parallel for private(i, j, k)
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        ans[i][j] += mat1[i][j] * mat2[k][j];
      }
    }
  }
}

// Parallel Strassen's algorithm
void strassen(Matrix &mat1, Matrix &mat2, Matrix &ans, int size) {
  if (size < AD_HOC) {
    multiply(mat1, mat2, ans, size);
    return;
  }

  int new_size = size / 2;
  std::vector<int> in_vec(new_size, 0);
  Matrix a11(new_size, in_vec), a12(new_size, in_vec), a21(new_size, in_vec),
      a22(new_size, in_vec), b11(new_size, in_vec), b12(new_size, in_vec),
      b21(new_size, in_vec), b22(new_size, in_vec), c11(new_size, in_vec),
      c12(new_size, in_vec), c21(new_size, in_vec), c22(new_size, in_vec),
      p1(new_size, in_vec), p2(new_size, in_vec), p3(new_size, in_vec),
      p4(new_size, in_vec), p5(new_size, in_vec), p6(new_size, in_vec),
      p7(new_size, in_vec), s1(new_size, in_vec), s2(new_size, in_vec),
      s3(new_size, in_vec), s4(new_size, in_vec), s5(new_size, in_vec),
      s6(new_size, in_vec), s7(new_size, in_vec), s8(new_size, in_vec),
      s9(new_size, in_vec), s10(new_size, in_vec), temp1(new_size, in_vec),
      temp2(new_size, in_vec);

  for (int i = 0; i < new_size; i++) {
    for (int j = 0; j < new_size; j++) {
      a11[i][j] = mat1[i][j];
      a12[i][j] = mat1[i][j + new_size];
      a21[i][j] = mat1[i + new_size][j];
      a22[i][j] = mat1[i + new_size][j + new_size];

      b11[i][j] = mat2[i][j];
      b12[i][j] = mat2[i][j + new_size];
      b21[i][j] = mat2[i + new_size][j];
      b22[i][j] = mat2[i + new_size][j + new_size];
    }
  }

#pragma omp parallel
  {
#pragma omp single
    {
      // M1 = (A11 + A22)*(B11 + B22)
#pragma omp task
      {
        add(a11, a22, s5, new_size);
        add(b11, b22, s6, new_size);
        strassen(s5, s6, p5, new_size);
      }

      // M2 = (A21 + A22)*B11
#pragma omp task
      {
        add(a21, a22, s3, new_size);
        strassen(s3, b11, p3, new_size);
      }

      // M3 = A11*(B12 - B22)
#pragma omp task
      {
        subtract(b12, b22, s1, new_size);
        strassen(a11, s1, p1, new_size);
      }

      // M4 = A22*(B21 - B11)
#pragma omp task
      {
        subtract(b21, b11, s4, new_size);
        strassen(a22, s4, p4, new_size);
      }

      // M5 = (A11 + A12)*B22
#pragma omp task
      {
        add(a11, a12, s2, new_size);
        strassen(s2, b22, p2, new_size);
      }

      // M6 = (A21 + A11)*(B11 - B12)
#pragma omp task
      {
        add(a21, a11, s9, new_size);
        subtract(b11, b12, s10, new_size);
        strassen(s9, s10, p7, new_size);
      }

      // M7 = (A12 - A22)*(B21 + B22)
#pragma omp task
      {
        subtract(a12, a22, s7, new_size);
        add(b21, b22, s8, new_size);
        strassen(s7, s8, p6, new_size);
      }
    }

#pragma omp taskwait
#pragma omp single
    {
#pragma omp task
      add(p1, p2, c12, new_size);
#pragma omp task
      add(p3, p4, c21, new_size);
#pragma omp task
      {
        add(p5, p4, temp1, new_size);
        add(temp1, p6, temp2, new_size);
        subtract(temp2, p2, c11, new_size);
      }
#pragma omp task
      {
        subtract(p5, p1, temp1, new_size);
        add(temp1, p3, temp2, new_size);
        add(temp2, p7, c22, new_size);
      }
    }
  }

  for (int i = 0; i < new_size; i++) {
    for (int j = 0; j < new_size; j++) {
      ans[i][j] = c11[i][j];
      ans[i][j + new_size] = a12[i][j];
      ans[i + new_size][j] = a21[i][j];
      ans[i + new_size][j + new_size] = a22[i][j];
    }
  }
}

int main() {

#ifdef _OPENMP
  omp_set_num_threads(NUM_THREADS);
#endif

  Matrix mat1;
  mat1.resize(SIZE);
  for (int i = 0; i < SIZE; i++) {
    mat1[i].resize(SIZE);
  }
  Matrix mat2;
  mat2.resize(SIZE);
  for (int i = 0; i < SIZE; i++) {
    mat2[i].resize(SIZE);
  }

  read_matrix(mat1, "mat1.txt");
  read_matrix(mat2, "mat2.txt");

  Matrix answer;
  answer.resize(SIZE);
  for (int i = 0; i < SIZE; i++) {
    answer[i].resize(SIZE);
  }

  float total_time = 0;

  for (int i = 0; i < TIMES; i++) {
#ifdef _OPENMP
    float t1 = omp_get_wtime();

#else
    clock_t st = clock();
#endif

    strassen(mat1, mat2, answer, SIZE);

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
  std::cout << "matrix size: " << SIZE << " * " << SIZE << std::endl;
  std::cout << "ad-hoc size: " << AD_HOC << std::endl;
  std::cout << "times of run: " << TIMES << std::endl;
}