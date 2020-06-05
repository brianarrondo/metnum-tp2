#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
    unsigned int n_neighbors;
    Matrix X_train;
    Matrix y_train;
};
