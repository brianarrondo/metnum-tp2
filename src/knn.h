#pragma once

#include "types.h"


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors, bool with_pca);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);
private:
	unsigned int n_neighbors;
	bool with_pca;
};
