#include <algorithm>
//#include <chrono>
#include <iostream>
#include <utility>
#include "knn.h"
#include "pca.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int neighbors, bool pca)
{
    n_neighbors = neighbors;
    with_pca = pca;
}

void KNNClassifier::fit(Matrix X, Matrix y)
{

    X_train = X;
    y_train = y;

/*
    // Hago un cambio de base en las matrices X e Y, quedandome las primeras n componentes
    unsigned int components = 25;
    auto pca = PCA(components);
    pca.fit(X);
    pca.fit(y);
*/
}

Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        Vector x = X.row(k);

        ret(k) = 0;
        auto min_distance = (x - X_train.row(0)).norm();

        for (unsigned i = 1; i < X_train.rows(); ++i)
        {
            auto distance = (x - X_train.row(i)).norm();
            if (distance < min_distance) {
                ret(k) = y_train.row(i)[0];
                min_distance = distance;
            }
        }
    }

    return ret;
}
