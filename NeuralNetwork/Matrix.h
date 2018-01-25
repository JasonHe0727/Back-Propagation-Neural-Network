#ifndef MATRIX_H
#define MATRIX_H
#include <initializer_list>
#include <iostream>
#include <vector>
using std::initializer_list;
using std::ostream;
using std::vector;
class Matrix;
class Matrix
{
public:
    Matrix();
    Matrix(size_t nRows, size_t nCols);
    Matrix(size_t nRows, size_t nCols, initializer_list<double> values);
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);

    double& operator()(size_t i, size_t j);
    friend ostream& operator<<(ostream& out, Matrix& matrix);
    void Display();
    size_t RowCount();
    size_t ColumnCount();

    void RandomSet();

    double SumOfRow(size_t i);
    double SumOfColumn(size_t j);

private:
    size_t nRows;
    size_t nCols;
    double* items;
};
#endif // MATRIX_H
