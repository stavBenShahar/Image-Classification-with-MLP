// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
using std::ostream;
using std::istream;
/**
 * @struct matrix_dims
 * @brief Matrix dimensions container. Used in MlpNetwork.h and main.cpp
 */
typedef struct matrix_dims
{
    int rows, cols;
} matrix_dims;

class Matrix
{

 public:
  Matrix ();
  /**
   * Constructs a matrix of size rows X cols and init all the elements to 0.
   * @param rows - Amount of rows of the Matrix
   * @param cols - Amount of cols of the Matrix.
   */
  Matrix (int rows, int cols);

  /**
   * Copy constructor, constructs a matrix from another Matrix.
   * @param m - The matrix the method copies.
   */
  Matrix (Matrix const &input_matrix);

  ~Matrix ();

  int get_rows () const
  { return m_nRows; }

  int get_cols () const
  { return m_nCols; }

  /**
   * Transforms a matrix into its transpose matrix.
   */
  Matrix &transpose ();

  /**
   * Transforms a matrix into a column vector.
   */
  Matrix &vectorize ();

  /**
   * Prints matrix elements, no return value.
   */
  void plain_print () const;

  /**
   *
   * @param m - the other Matrix.
   * @return: s a matrix which is the elementwise multiplication(Hadamard
   * product) of this matrix and another matrix m.
   */
  Matrix dot (Matrix const &m) const;

  /**
   *
   * @return: The Frobenius norm of the given matrix.
   */
  float norm ();

  /******************* Operators *******************/

  Matrix operator+ (const Matrix &input_matrix) const;
  Matrix &operator= (const Matrix &input_matrix);
  Matrix operator* (const Matrix &input_matrix) const;
  Matrix operator* (float &scalar) const;
  friend Matrix operator* (const float &scalar, const Matrix &input_matrix);
  void operator+= (const Matrix &input_matrix);
  float operator() (int row_num, int col_num) const;
  float &operator() (int row_num, int col_num);
  float operator[] (int value_coordination) const;
  float &operator[] (int value_coordination);
  friend ostream &operator<< (ostream &stream, const Matrix &input_matrix);
  friend istream &operator>> (istream &stream, Matrix &input_matrix);

 private:
  int m_nRows;
  int m_nCols;
  float *m_Data;

  float norm () const;
};

#endif //MATRIX_H