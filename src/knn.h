#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int neighbors, bool pca);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
    unsigned int n_neighbors;
    bool with_pca;
    Matrix X_train;
    Matrix y_train;
};
