#ifndef DENSE_H
#define DENSE_H

#include "Activation.h"
using activation::activation_func;

class Dense
{

 private:
  Matrix _weight;
  Matrix _bias;
  activation_func _activation;

 public:
  //Constructor
  Dense (Matrix &weight, Matrix &bias, activation_func activation);
  //Destructor
  Matrix get_weights () const
  { return this->_weight; }

  Matrix get_bias () const
  { return this->_bias; }

  activation_func get_activation () const
  { return this->_activation; }
  /**
   * Applies the layer on input matrix.
   * @param input_matrix - The matrix that was created in the previous layer.
   * @return A new matrix that was created from the current layer of the
   * Mlp network.
   */
  Matrix operator() (Matrix &input_matrix) const;

};

#endif //DENSE_H
