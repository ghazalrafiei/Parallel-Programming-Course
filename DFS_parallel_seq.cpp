#include <atomic>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
const int MAX_N = 100;

void parallel_dfs(int graph[][MAX_N], int size, int root) {
  std::atomic<bool> marked[MAX_N];
  for (int i = 0; i < size; i++) {
    marked[i] = false;
  }
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < size; i++) {
    std::stack<int> unvisited_vertices;
    marked[i] = true;
#pragma omp critical
    unvisited_vertices.push(i);

    while (unvisited_vertices.size()) {
#pragma omp critical
      {
        root = unvisited_vertices.top();
        unvisited_vertices.pop();
      }
      for (int adj = 0; adj < size; adj++) {
        if (graph[root][adj] && !marked[adj]) {
          marked[adj] = true;
#pragma omp critical
          unvisited_vertices.push(adj);
        }
      }
    }
  }
}

void sequential_dfs(int graph[][MAX_N], int size, int root) {
  std::atomic<bool> marked[MAX_N];
  std::stack<int> unvisited_vertices;
  for (int i = 0; i < size; i++) {
    marked[i] = false;
  }
  marked[root] = true;
  unvisited_vertices.push(root);

  while (unvisited_vertices.size()) {
    root = unvisited_vertices.top();
    unvisited_vertices.pop();
    for (int i = 0; i < size; i++) {
      if (graph[root][i] && !marked[i]) {
        unvisited_vertices.push(i);
        marked[i] = true;
      }
    }
  }
}

int main() {
  int n;
  printf("Size of graph = ");
  scanf("%d", &n);

  int graph[MAX_N][MAX_N];
  printf("Enter the graph entries in adjacency matrix form:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      scanf("%d", &graph[i][j]);
    }
  }
  const int times = 10'000;
  float sum = 0;
  for (int i = 0; i < times; i++) {
    clock_t st = clock();
    // sequential_dfs(graph, n, 0);
    parallel_dfs(graph, n, 0);

    sum += float(clock() - st) / CLOCKS_PER_SEC;
  }

  printf("average spent time = %f\n", sum / times);

  return 0;
}
