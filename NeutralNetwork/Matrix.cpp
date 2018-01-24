#include "Matrix.h"
#include <iostream>
#include <stdlib.h>
using namespace std;
Matrix::Matrix() : nRows{0}, nCols{0}, items{nullptr}
{
}
Matrix::Matrix(size_t nRows, size_t nCols)
    : nRows{nRows}, nCols{nCols}, items{new double[nRows * nCols]}
{
    size_t size = nRows * nCols;
    for (size_t i = 0; i < size; i++)
    {
        items[i] = 0.0;
    }
}
Matrix::Matrix(size_t nRows, size_t nCols, initializer_list<double> values)
    : nRows{nRows}, nCols{nCols}, items{new double[nRows * nCols]}
{
    size_t i = 0;
    for (double item : values)
    {
        items[i] = item;
        i++;
    }
}

Matrix::~Matrix()
{
    delete[] items;
}

Matrix::Matrix(const Matrix& other)
{
    nRows = other.nRows;
    nCols = other.nCols;
    size_t size = nRows * nCols;
    items = new double[size];
    for (size_t i = 0; i < size; i++)
    {
        items[i] = other.items[i];
    }
}

Matrix& Matrix::operator=(const Matrix& other)
{
    nRows = other.nRows;
    nCols = other.nCols;
    size_t size = nRows * nCols;
    items = new double[size];
    for (size_t i = 0; i < size; i++)
    {
        items[i] = other.items[i];
    }
    return *this;
}

Matrix::Matrix(Matrix&& other)
{
    nRows = other.nRows;
    nCols = other.nCols;
    items = other.items;
    other.items = nullptr;
}

Matrix& Matrix::operator=(Matrix&& other)
{
    nRows = other.nRows;
    nCols = other.nCols;
    items = other.items;
    other.items = nullptr;
    return *this;
}

double& Matrix::operator()(size_t i, size_t j)
{
    return items[i * nCols + j];
}

ostream& operator<<(ostream& out, Matrix& matrix)
{
    out << matrix.nRows << " x " << matrix.nCols << endl;
    for (size_t i = 0; i < matrix.nRows; i++)
    {
        for (size_t j = 0; j < matrix.nCols; j++)
        {
            out << matrix.items[i * matrix.nCols + j] << "  ";
        }
        out << endl;
    }
    return out;
}

void Matrix::Display()
{
    cout << nRows << " x " << nCols << endl;
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            cout << items[i * nCols + j] << "  ";
        }
        cout << endl;
    }
}

size_t Matrix::RowCount()
{
    return nRows;
}

size_t Matrix::ColumnCount()
{
    return nCols;
}

void Matrix::RandomSet()
{
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            items[i * nCols + j] = rand() / double(RAND_MAX);
        }
    }
}

double Matrix::SumOfRow(size_t i)
{
    double sum = 0.0;
    for (size_t j = 0; j < nCols; j++)
    {
        sum = sum + items[i * nCols + j];
    }
    return sum;
}

double Matrix::SumOfColumn(size_t j)
{
    double sum = 0.0;
    for (size_t i = 0; i < nRows; i++)
    {
        sum = sum + items[i * nCols + j];
    }
    return sum;
}
