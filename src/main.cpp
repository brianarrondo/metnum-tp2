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

pair<Matrix, Matrix> read_input_params(int argc, char *argv[]) {
	if (argc < 8) {
		cout << "Debe ingresar ocho parametros de entrada" << endl;
		exit(1);
	}

	string method = argv[2];
	string train_set = argv[4];
	string test_set = argv[6];
	string classif = argv[8];

	cout << endl;
	cout<< "Method: " << method << endl;
	cout<< "Train_set: " << train_set << endl;
	cout<< "Test_set: " << test_set << endl;
	cout<< "Classif: " << classif << endl;
	cout << endl;

	Matrix train = read_csv_to_matrix(train_set);
	Matrix test = read_csv_to_matrix(test_set);

	cout << "DIM. MATRIZ ENTRENAMIENTO: " << train.rows() << "x" << train.cols() << endl;
	cout << "DIM. MATRIZ TEST: " << test.rows() << "x" << test.cols() << endl;

	return make_pair(train, test);
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

int main(int argc, char** argv) {

	auto t0 = clock();
	pair<Matrix, Matrix> m = read_input_params(argc, argv);
	auto t1 = clock();

	double time = (double(t1 - t0)/CLOCKS_PER_SEC);
	cout << "Execution Time (read input params): " << time << endl;

	Matrix X = m.first;
	// cout << "X: \n" << X << endl;

	Matrix y_train = get_first_column(X);
	Matrix X_train = remove_first_column(X);

	// cout << "X_train: \n" << X_train << endl;
	// cout << "y_train: \n" << y_train << endl;

	auto knn = KNNClassifier(10, false);
	knn.fit(X_train, y_train);

	Matrix test(28, 28);
	test << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,152,71,51,51,21,41,51,51,51,51,113,193,152,30,0,0,0,0,0,0,0,0,0,0,0,0,0,122,253,252,253,252,223,243,253,252,253,252,253,252,233,30,0,0,0,0,0,0,0,0,0,0,0,0,0,123,102,41,102,102,102,102,102,102,102,162,254,253,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,102,0,0,0,0,0,0,0,0,183,253,212,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,142,0,0,0,0,0,0,0,11,213,254,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,243,102,0,0,0,0,0,0,0,51,252,172,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,223,102,0,0,0,0,0,0,0,214,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,0,0,0,0,0,0,0,253,252,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,254,253,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,102,253,171,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,163,254,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,253,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,253,254,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,252,253,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,253,254,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,252,213,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,152,253,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,233,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,255,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,253,212,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;

	cout << test << endl;
	cout << "Predict: " << endl << knn.predict(test) << endl;

	return 0;
}
