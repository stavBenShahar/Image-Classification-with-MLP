#include "MlpNetwork.h"

MlpNetwork::MlpNetwork(Matrix weights[], Matrix biases[]) {
  for (int i = 0; i < MLP_SIZE; ++i) {
    _layers[i] = Dense(weights[i], biases[i], (i == MLP_SIZE - 1) ? activation::softmax : activation::relu);
  }
}

digit MlpNetwork::operator()(const Matrix &input) const {
  Matrix result = input;
  for (const auto &layer : _layers) {
    result = layer(result);
  }
  return getHighestProbabilityDigit(result);
}

digit MlpNetwork::getHighestProbabilityDigit(const Matrix &output) const {
  digit maxDigit = {0, output[0]};
  for (int i = 1; i < OUTPUT_VECTOR_SIZE; ++i) {
    if (output[i] > maxDigit.probability) {
      maxDigit.value = i;
      maxDigit.probability = output[i];
    }
  }
  return maxDigit;
}
