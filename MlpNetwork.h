//MlpNetwork.h

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include "Dense.h"
#define FINAL_LAYER_INDEX MLP_SIZE - 1
#define MLP_SIZE 4
#define OUTPUT_VECTOR_SIZE 10

/**
 * @struct digit
 * @brief Identified (by Mlp network) digit with
 *        the associated probability.
 * @var value - Identified digit value
 * @var probability - identification probability
 */
typedef struct digit
{
    unsigned int value;
    float probability;
} digit;

const matrix_dims img_dims = {28, 28};
const matrix_dims weights_dims[] = {{128, 784},
                                    {64,  128},
                                    {20,  64},
                                    {10,  20}};
const matrix_dims bias_dims[] = {{128, 1},
                                 {64,  1},
                                 {20,  1},
                                 {10,  1}};

class MlpNetwork
{
 private:
  Matrix *_weights;
  Matrix *_biases;
  /**
   *
   * @param input_vector - The vector that was created from the last layer
   * of the network.
   * @return The digit with the highest probability in the input_vector.
   */
  digit get_highest_probability_digit (Matrix &input_vector) const;

 public:
  //Constructor
  MlpNetwork (Matrix *weights, Matrix *biases);
  /**
   * Applies the entire network on the input_matrix.
   * @param input_matrix - The matrix that represents the input image.
   * @return The digit with the highest probability.
   */
  digit operator() (Matrix &input_matrix) const;
  digit getHighestProbabilityDigit (const Matrix &output) const;
};
#endif // MLPNETWORK_H