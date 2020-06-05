#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int n_components;
    Matrix cov_matrix;
    Matrix eigen_vectors;
};
