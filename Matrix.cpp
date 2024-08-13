#include "Matrix.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>

#define INVALID_INDEX -1
#define DEFAULT_SIZE 1
#define LENGTH_ERROR_MSG "Error: Invalid matrix size."
#define OUT_OF_RANGE_MSG "Error: Index out of range."
#define INVALID_PATH_MSG "Error: Invalid file path."
#define INVALID_FILE_SIZE_MSG "Error: Invalid file size."

Matrix::Matrix() : m_nRows(1), m_nCols(1), m_Data(new float[1]{0.0}) {}

Matrix::Matrix(int rows, int cols) : m_nRows(rows), m_nCols(cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::length_error(LENGTH_ERROR_MSG);
  }
  m_Data = new float[rows * cols]();
}

Matrix::Matrix(const Matrix &other) : m_nRows(other.m_nRows), m_nCols(other.m_nCols), m_Data(new float[other.m_nRows * other.m_nCols]) {
  std::memcpy(m_Data, other.m_Data, m_nRows * m_nCols * sizeof(float));
}

Matrix::~Matrix() {
  delete[] m_Data;
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this == &other) {
    return *this;
  }
  delete[] m_Data;
  m_nRows = other.m_nRows;
  m_nCols = other.m_nCols;
  m_Data = new float[m_nRows * m_nCols];
  std::memcpy(m_Data, other.m_Data, m_nRows * m_nCols * sizeof(float));
  return *this;
}

Matrix &Matrix::transpose() {
  Matrix transposed(m_nCols, m_nRows);
  for (int i = 0; i < m_nRows; ++i) {
    for (int j = 0; j < m_nCols; ++j) {
      transposed(j, i) = (*this)(i, j);
    }
  }
  *this = transposed;
  return *this;
}

Matrix &Matrix::vectorize() {
  m_nRows *= m_nCols;
  m_nCols = DEFAULT_SIZE;
  return *this;
}

void Matrix::plain_print() const {
  for (int i = 0; i < m_nRows; ++i) {
    for (int j = 0; j < m_nCols; ++j) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

Matrix Matrix::dot(const Matrix &other) const {
  if (m_nRows != other.m_nRows || m_nCols != other.m_nCols) {
    throw std::length_error(LENGTH_ERROR_MSG);
  }
  Matrix result(m_nRows, m_nCols);
  for (int i = 0; i < m_nRows * m_nCols; ++i) {
    result.m_Data[i] = m_Data[i] * other.m_Data[i];
  }
  return result;
}

float Matrix::norm() const {
  float sum = 0.0;
  for (int i = 0; i < m_nRows * m_nCols; ++i) {
    sum += m_Data[i] * m_Data[i];
  }
  return std::sqrt(sum);
}

Matrix Matrix::operator+(const Matrix &other) const {
  if (m_nRows != other.m_nRows || m_nCols != other.m_nCols) {
    throw std::length_error(LENGTH_ERROR_MSG);
  }
  Matrix result(m_nRows, m_nCols);
  for (int i = 0; i < m_nRows * m_nCols; ++i) {
    result.m_Data[i] = m_Data[i] + other.m_Data[i];
  }
  return result;
}

Matrix Matrix::operator*(const Matrix &other) const {
  if (m_nCols != other.m_nRows) {
    throw std::length_error(LENGTH_ERROR_MSG);
  }
  Matrix result(m_nRows, other.m_nCols);
  for (int i = 0; i < m_nRows; ++i) {
    for (int j = 0; j < other.m_nCols; ++j) {
      result(i, j) = 0;
      for (int k = 0; k < m_nCols; ++k) {
        result(i, j) += (*this)(i, k) * other(k, j);
      }
    }
  }
  return result;
}

Matrix Matrix::operator*(float scalar) const {
  Matrix result(m_nRows, m_nCols);
  for (int i = 0; i < m_nRows * m_nCols; ++i) {
    result.m_Data[i] = m_Data[i] * scalar;
  }
  return result;
}

Matrix operator*(float scalar, const Matrix &mat) {
  return mat * scalar;
}

void Matrix::operator+=(const Matrix &other) {
  if (m_nRows != other.m_nRows || m_nCols != other.m_nCols) {
    throw std::length_error(LENGTH_ERROR_MSG);
  }
  for (int i = 0; i < m_nRows * m_nCols; ++i) {
    m_Data[i] += other.m_Data[i];
  }
}

float &Matrix::operator()(int row, int col) {
  if (row < 0 || col < 0 || row >= m_nRows || col >= m_nCols) {
    throw std::out_of_range(OUT_OF_RANGE_MSG);
  }
  return m_Data[row * m_nCols + col];
}

float Matrix::operator()(int row, int col) const {
  if (row < 0 || col < 0 || row >= m_nRows || col >= m_nCols) {
    throw std::out_of_range(OUT_OF_RANGE_MSG);
  }
  return m_Data[row * m_nCols + col];
}

float &Matrix::operator[](int idx) {
  if (idx < 0 || idx >= m_nRows * m_nCols) {
    throw std::out_of_range(OUT_OF_RANGE_MSG);
  }
  return m_Data[idx];
}

float Matrix::operator[](int idx) const {
  if (idx < 0 || idx >= m_nRows * m_nCols) {
    throw std::out_of_range(OUT_OF_RANGE_MSG);
  }
  return m_Data[idx];
}

std::ostream &operator<<(std::ostream &os, const Matrix &mat) {
  for (int i = 0; i < mat.m_nRows; ++i) {
    for (int j = 0; j < mat.m_nCols; ++j) {
      os << (mat(i, j) > 0.1 ? "**" : "  ");
    }
    os << std::endl;
  }
  return os;
}

std::istream &operator>>(std::istream &is, Matrix &mat) {
  if (!is) {
    throw std::runtime_error(INVALID_PATH_MSG);
  }

  long int fileSize = mat.m_nCols * mat.m_nRows * sizeof(float);
  is.seekg(0, std::ios::end);
  if (is.tellg() != fileSize) {
    throw std::runtime_error(INVALID_FILE_SIZE_MSG);
  }

  is.seekg(0, std::ios::beg);
  is.read(reinterpret_cast<char *>(mat.m_Data), fileSize);
  return is;
}
