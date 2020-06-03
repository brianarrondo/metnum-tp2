#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;

PCA::PCA(unsigned int components)
{
    n_components = components;
}

void PCA::fit(Matrix X)
{
    // Calculo la matriz de covarianza de la matriz de entrenamiento
    Matrix A = X.rowwise() - X.colwise().mean();
    Matrix cov_matrix = (A.adjoint() * A) / double(X.rows() - 1);

    // A partir de la matriz de covarianza, obtengo los primeros n autovectores y autovalores
    pair<Vector, Matrix> eigen_values_vectors = get_first_eigenvalues(cov_matrix, n_components);

    // Obtengo los autovectores de la tupla
    Matrix eigen_vectors = eigen_values_vectors.second;

    // Creo una matriz auxiliar para retornar
    Matrix Ret;

    // Cambio de base a la matriz de entrenamiento
    Ret = X * eigen_vectors;

    // Retorno la matriz con el cambio de base
    X = Ret;
}


MatrixXd PCA::transform(Matrix X)
{

  throw std::runtime_error("Sin implementar");
}
