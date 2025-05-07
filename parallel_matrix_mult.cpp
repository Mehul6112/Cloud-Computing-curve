#include <iostream>
#include <omp.h>

const int N = 1000;

void matrix_multiply(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    static double A[N][N], B[N][N], C[N][N];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }

    double start = omp_get_wtime();
    matrix_multiply(A, B, C);
    double end = omp_get_wtime();

    std::cout << "Matrix multiplication completed in " << (end - start) << " seconds.\n";

    return 0;
}

/*To run this code:
execute "g++ -fopenmp -O2 parallel_matrix_mult.cpp -o parallel_matrix_mult" on terminal
then execute "./matrix_mult"

*/
