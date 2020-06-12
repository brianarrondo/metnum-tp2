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
        Vector b_old = b;
        Vector Xb = X * b;
        b = Xb / Xb.norm();

        double cos_angle = b.dot(b_old);
        if ((1 - eps) < cos_angle && cos_angle <= 1) {
            i = num_iter;
        }
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
        pair<double, Vector> eigen_value_vector = power_iteration(A, num_iter, epsilon);
        double eigen_value = eigen_value_vector.first;
        Vector eigen_vector = eigen_value_vector.second;

        // Pusheo el nuevo autovalor y autovector
        eigvalues[i] = eigen_value;
        eigvectors.col(i) = eigen_vector;

        // Reinterpreto A (deflacion)
        A = A - (eigen_value * (eigen_vector * eigen_vector.transpose()));
    }

    return make_pair(eigvalues, eigvectors);
}  
