//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
#include "knn.h"
#include <fstream>
#include <string>

using namespace std;

vector<double> split(string line, string delimiter) {
    vector<double> V;
    string token = line;
    unsigned int token_end = 0;

    while (token_end < line.length()) {
        token_end = token.find(delimiter);
        double pixel_intensity = stod(token.substr(0, token_end));
        token = token.substr(token_end + 1);
        V.push_back(pixel_intensity);
    }

    return V;
}

Matrix read_csv_to_matrix(string input_csv) {
    ifstream file;

    file.open(input_csv);
    if (!file) {
        cout << "Error al abrir el archivo: " << input_csv << endl;
        exit(1);
    }

    Matrix A;
    string line;
    string delimiter = ",";
    unsigned int index = 0;

    getline(file, line);

    while (!file.eof()) {
        getline(file, line);

        if (line.length() > 0) {
            vector<double> image_vector = split(line, delimiter);
            A.conservativeResize(index + 1, image_vector.size());

            Vector V = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(image_vector.data(), image_vector.size());
            A.row(index) = V.transpose();

            index++;
        }
    }

    file.close();

    return A;
}

tuple<string , Matrix, Matrix, string> read_input_params(int argc, char *argv[]) {
    if (argc < 8) {
        cout << "Debe ingresar ocho parametros de entrada" << endl;
        exit(1);
    }

    string method = argv[2];
    string train_set = argv[4];
    string test_set = argv[6];
    string classif = argv[8];

    if (method != "0" && method != "1") {
        cout << "Method ID invalid. Use 0 (Knn) or 1 (PCA + Knn)." << endl;
        exit(1);
    }

    cout << endl;
    cout<< "Method: " << (method == "0" ? "Knn" : "PCA + Knn") << endl;
    cout<< "Train_set: " << train_set << endl;
    cout<< "Test_set: " << test_set << endl;
    cout<< "Classif: " << classif << endl;
    cout << endl;

    Matrix train = read_csv_to_matrix(train_set);
    Matrix test = read_csv_to_matrix(test_set);

    return make_tuple(method, train, test, classif);
}

Matrix remove_first_column(Matrix X) {
    unsigned int last = X.cols() - 1;
    X.col(0).swap(X.col(last));
    X.conservativeResize(X.rows(), X.cols() - 1);

    for (unsigned int i = 0; i < X.cols() -1; ++i) {
        X.col(i).swap(X.col(i+1));
    }

    return X;
}

Vector get_first_column(Matrix X) {
    return X.col(0);
}

void print_csv_output(Vector V, string output_file) {
    FILE *fp = fopen(output_file.c_str(), "w");

    fprintf(fp, "ImageId,Label\n");
    if (fp != NULL) {
        for(int i = 0; i < V.rows(); i++){
            fprintf(fp, "%u,%f\n", i, V[i]);
        }
    } else {
        cout << "Error al abrir el archivo: " << output_file << endl;
        exit(1);
    }

    fclose(fp);
}

int main(int argc, char** argv) {

    auto t0 = clock();
    tuple<string , Matrix, Matrix, string> args = read_input_params(argc, argv);
    auto t1 = clock();

    double time = (double(t1 - t0)/CLOCKS_PER_SEC);
    cout << "Execution Time (read input params): " << time << " seconds" << endl;
    cout << endl;

    string method = get<0>(args);
    Matrix train_set = get<1>(args);
    Matrix test_set = get<2>(args);
    string classif = get<3>(args);

    cout << "Dimension train_set: " << train_set.rows() << "x" << train_set.cols() << endl;
    cout << "Dimension test_set: " << test_set.rows() << "x" << test_set.cols() << endl;
    cout << endl;

    Matrix y_train = get_first_column(train_set);
    Matrix X_train = remove_first_column(train_set);

    int neighbors = 10;
    int components = 10;
    Vector predict;
    if (method == "0") {
        auto knn = KNNClassifier(neighbors);
        knn.fit(X_train, y_train);
        predict = knn.predict(test_set);

        cout << "Knn params: " << endl;
        cout << "   - Neighbors: " << neighbors << endl;
        cout << endl;

        cout << "Knn Predict: " << endl;
        for (unsigned i = 0; i < predict.rows(); ++i)
        {
            cout << predict(i);
            if (i+1 < predict.rows()) {
                cout << ", ";
            }
        }
        cout << endl;

    } else if (method == "1") {
        auto knn = KNNClassifier(neighbors);
        auto pca = PCA(components);

        pca.fit(X_train);
        Matrix X_train_transformed = pca.transform(X_train);
        knn.fit(X_train_transformed, y_train);
        Matrix test_set_transformed = pca.transform(test_set);

        predict = knn.predict(test_set_transformed);

        cout << "PCA + Knn params: " << endl;
        cout << "   - Neighbors: " << neighbors << endl;
        cout << "   - Components: " << components << endl;
        cout << endl;

        cout << "PCA + Knn Predict: " << endl;
        for (unsigned i = 0; i < predict.rows(); ++i)
        {
            cout << predict(i);
            if (i+1 < predict.rows()) {
                cout << ", ";
            }
        }
        cout << endl;
    } else {
        cout << "Method ID invalid. Use 0 (Knn) or 1 (PCA + Knn)." << endl;
    }

    print_csv_output(predict, classif);

    return 0;
}
