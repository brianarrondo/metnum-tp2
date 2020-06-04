//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"
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
	vector<vector<double>> M(42000);
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

void read_input_params(int argc, char *argv[]) {
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
}

int main(int argc, char** argv) {

	auto t0 = clock();
	read_input_params(argc, argv);
	auto t1 = clock();

	double time = (double(t1 - t0)/CLOCKS_PER_SEC);
	cout << "Execution Time (read input params): " << time << endl;

	return 0;
}
