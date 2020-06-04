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

void read_input_params(int argc, char *argv[]) {
	if (argc < 8) {
		std::cout << "Debe ingresar ocho parametros de entrada" << std::endl;
		exit(1);
	}

	string method = argv[2];
	string train_set = argv[4];
	string test_set = argv[6];
	string classif = argv[8];

	std::cout << std::endl;
	std::cout<< "Method: " << method << std::endl;
	std::cout<< "Train_set: " << train_set << std::endl;
	std::cout<< "Test_set: " << test_set << std::endl;
	std::cout<< "Classif: " << classif << std::endl;
	std::cout << std::endl;

	ifstream file;

	file.open(train_set);
	if (!file) {
		cout << "Error al abrir el archivo: " << train_set << endl;
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

	// PROBANDO AUTOVECTORES Y AUTOVALORES
	cout << "MATRIZ DE : " << A.rows() << "x" << A.cols() << endl;
	cout << endl;

	auto eigen_values_vectors = get_first_eigenvalues(A, 6);
	Vector eigen_values = eigen_values_vectors.first;
	Matrix eigen_vectors = eigen_values_vectors.second;

	std::cout << "MATRIZ ORIGINAL: \n" << A << std::endl << endl;
	std::cout << "AUTOVALORES : \n" << eigen_values << std::endl << endl;
	std::cout << "AUTOVECTORES : \n" << eigen_vectors << std::endl << endl;
}

int main(int argc, char** argv){
	read_input_params(argc, argv);

	return 0;
}
