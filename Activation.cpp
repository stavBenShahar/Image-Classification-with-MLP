#include "Activation.h"
#include <algorithm>
#include <cmath>

Matrix activation::relu(const Matrix &input) {
  Matrix output(input.get_rows(), input.get_cols());
  std::transform(input.begin(), input.end(), output.begin(), [](float val) { return std::max(0.0f, val); });
  return output;
}

Matrix activation::softmax(const Matrix &input) {
  Matrix output(input.get_rows(), input.get_cols());
  float sum = 0.0f;
  std::transform(input.begin(), input.end(), output.begin(), [&sum](float val) {
      float expVal = std::exp(val);
      sum += expVal;
      return expVal;
  });
  return output * (1.0f / sum);
}
