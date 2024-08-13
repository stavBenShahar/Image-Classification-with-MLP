#include "Matrix.h"
#include <cmath>
#ifndef ACTIVATION_H
#define ACTIVATION_H
using std::exp;

namespace activation
{
    typedef Matrix (*activation_func) (Matrix &);
    /**
     * The relu turns the input values of the matrix to max {0,value}.
     * @return A matrix that is the function relu on the input matrix.
     */
    Matrix relu (const Matrix &input);
    /**
     * The softmax turns the input into a small probability, and if an input
     * is large, then it turns it into a large probability, but it will
     * always remain between 0 and 1.
     * @return A matrix that is the function softmax on the input matrix.
     */
    Matrix softmax (const Matrix &input);


}
#endif //ACTIVATION_H