#include <algorithm>
//#include <chrono>
#include <iostream>
#include <utility>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int neighbors)
{
    n_neighbors = neighbors;
}

void KNNClassifier::fit(Matrix X, Matrix y)
{
    X_train = X;
    y_train = y;
}

Vector KNNClassifier::predict(Matrix X)
{
    // Creamos vector columna a devolver
    auto ret = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k)
    {
        Vector x = X.row(k).transpose();

        vector<pair<int, double>> digit_distances;

        for (unsigned i = 0; i < X_train.rows(); ++i)
        {
            int digit = y_train.row(i)[0];
            auto distance = (x - X_train.row(i).transpose()).norm();

            digit_distances.emplace_back(digit, distance);
        }

        // Ordeno los digitos por distancia
        std::sort(digit_distances.begin(), digit_distances.end(), [](const std::pair<int, double> &left, const std::pair<int, double> &right) {
            return left.second < right.second;
        });

        // Me quedo con los n_neighbors mas cercanos
        digit_distances.erase(digit_distances.begin() + n_neighbors, digit_distances.end());

        // Votacion
        vector<int> vote(10);
        for (unsigned n = 0; n < n_neighbors; ++n)
        {
            vote[digit_distances[n].first] = vote[digit_distances[n].first] + 1;
        }

        // Indice del elemento mas grande, osea, con mas votos
        int maxElementIndex = std::max_element(vote.begin(), vote.end()) - vote.begin();

        ret(k) = maxElementIndex;
    }

    return ret;
}
