#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
    Vector b = Vector::Random(X.cols());
    double eigenvalue;

    for (int i = 0; i < num_iter; ++i)
    {
        Vector Xb = X * b;
        b = Xb / Xb.norm();
    }

    Vector Xb = X * b;

    // Calculamos el autovalor correspondiente al autovector
    eigenvalue = (double) (b.transpose() * Xb) / (b.squaredNorm());

    // Retornamos el autovalor y el autovector normalizado (No entiendo porque devolver b normalizado)
    return make_pair(eigenvalue, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (int i = 0; i < num; ++i)
    {
        // Obtengo un autovalor y autovector de X
        pair<double, Vector> eigen_value_vector = power_iteration(A);
        double eigen_value = eigen_value_vector.first;
        Vector eigen_vector = eigen_value_vector.second;

        // Pusheo el nuevo autovalor y autovector
        eigvalues[i] = eigen_value;
        eigvectors.col(i) = eigen_vector;

        // Reinterpreto A
        A = A - (eigen_value * (eigen_vector * eigen_vector.transpose()));
    }

    return make_pair(eigvalues, eigvectors);
}  
