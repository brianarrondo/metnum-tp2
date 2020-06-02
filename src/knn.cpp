#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include "pca.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors)
{
	neighbors = n_neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y)
{
	// Hago un cambio de base en las matrices X e Y, quedandome las primeras n componentes
	unsigned int n_components = 25;
	auto pca = PCA(n_components);
	pca.fit(X);
	pca.fit(y);
}


Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        ret(k) = 0;
    }

    return ret;
}
