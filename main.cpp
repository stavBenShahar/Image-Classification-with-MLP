#include "Matrix.h"
#include "Activation.h"
#include "Dense.h"
#include "MlpNetwork.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

#define QUIT "q"
#define INSERT_IMAGE_PATH "Please insert image path:"
#define ERROR_INVALID_PARAMETER "Error: Invalid parameters file for layer: "
#define ERROR_INVALID_INPUT "Error: Failed to retrieve input. Exiting.."
#define ERROR_INVALID_IMG "Error: Invalid image path or size: "

#define USAGE_ERR "Usage: mlp_network <weights> <biases>"
#define ARGS_COUNT (1 + MLP_SIZE * 2)
#define WEIGHTS_START_IDX 1
#define BIAS_START_IDX (WEIGHTS_START_IDX + MLP_SIZE)

/**
 * Prints program usage to stdout and checks the number of arguments.
 * @param argc number of arguments given in the program
 * @throw std::domain_error in case of wrong number of arguments
 */
void checkUsage(int argc) {
  if (argc != ARGS_COUNT) {
    throw std::domain_error(USAGE_ERR);
  }
  std::cout << USAGE_ERR << std::endl;
}

/**
 * Given a binary file path and a matrix, reads the content of the file into the matrix.
 * File must match matrix in size to read successfully.
 * @param filePath - path of the binary file to read
 * @param mat - matrix to read the file into.
 * @return boolean status
 *          true - success
 *          false - failure
 */
bool readFileToMatrix(const std::string &filePath, Matrix &mat) {
  std::ifstream is(filePath, std::ios::in | std::ios::binary);
  if (!is.is_open()) {
    return false;
  }

  long int matByteSize = static_cast<long int>(mat.get_cols() * mat.get_rows() * sizeof(float));
  is.seekg(0, std::ios_base::end);
  if (is.tellg() != matByteSize) {
    is.close();
    return false;
  }

  is.seekg(0, std::ios_base::beg);
  is >> mat;
  return is.good();
}

/**
 * Loads MLP parameters from weights & biases paths into arrays.
 * Throws an exception upon failure.
 * @param paths array of program arguments, expected to be MLP parameters paths.
 * @param weights array of matrices, weights[i] is the i-th layer weights matrix.
 * @param biases array of matrices, biases[i] is the i-th layer bias matrix (which is a vector).
 * @throw std::invalid_argument in case of problem with a certain argument
 */
void loadParameters(char *paths[], Matrix weights[], Matrix biases[]) {
  for (int i = 0; i < MLP_SIZE; ++i) {
    weights[i] = Matrix(weights_dims[i].rows, weights_dims[i].cols);
    biases[i] = Matrix(bias_dims[i].rows, bias_dims[i].cols);

    std::string weightsPath(paths[WEIGHTS_START_IDX + i]);
    std::string biasPath(paths[BIAS_START_IDX + i]);

    if (!readFileToMatrix(weightsPath, weights[i]) || !readFileToMatrix(biasPath, biases[i])) {
      throw std::invalid_argument(ERROR_INVALID_PARAMETER + std::to_string(i + 1));
    }
  }
}

/**
 * Command line interface for the MLP network.
 * Loops on: {Retrieve user input, Feed input to MLP network, Print image & network prediction}
 * Throws an exception on fatal errors.
 * @param mlp MlpNetwork to use for prediction.
 * @throw std::invalid_argument in case of problem with the user input path
 */
void mlpCli(MlpNetwork &mlp) {
  Matrix img(img_dims.rows, img_dims.cols);
  std::string imgPath;

  while (true) {
    std::cout << INSERT_IMAGE_PATH << std::endl;
    std::cin >> imgPath;
    if (!std::cin.good() || imgPath == QUIT) {
      break;
    }

    if (readFileToMatrix(imgPath, img)) {
      Matrix imgVec = img.vectorize();
      digit output = mlp(imgVec);
      std::cout << "Image processed:" << std::endl << img << std::endl;
      std::cout << "MLP result: " << output.value << " at probability: " << output.probability << std::endl;
    } else {
      throw std::invalid_argument(ERROR_INVALID_IMG + imgPath);
    }
  }
}

/**
 * Program's main entry point.
 * @param argc count of args
 * @param argv args values
 * @return program exit status code
 */
int main(int argc, char **argv) {
  try {
    checkUsage(argc);
    Matrix weights[MLP_SIZE];
    Matrix biases[MLP_SIZE];
    loadParameters(argv, weights, biases);
    MlpNetwork mlp(weights, biases);
    mlpCli(mlp);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
