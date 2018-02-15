#include "BPNeuralNetwork.hpp"
#include <cmath>
#include <iostream>
#include <random>
using namespace std;

BPNeuralNetwork::BPNeuralNetwork(int numOfInput, vector<int> numOfHidden,
                                 int numOfOutput)
    : input(numOfInput, 1)
    , hidden(numOfHidden.size())
    , output(numOfOutput, 1)
    , weights(numOfHidden.size() + 1)
    , biases(numOfHidden.size() + 1)
    , deltas(numOfHidden.size() + 1)
{
    if (numOfHidden.size() < 1)
    {
        throw "There should be at least one hidden layer";
    }
    else
    {
        weights.at(0) = Matrix<double>(numOfHidden.at(0), numOfInput);
        biases.at(0) = Matrix<double>::RowVector(numOfHidden.at(0));
        deltas.at(0) = Matrix<double>::RowVector(numOfHidden.at(0));

        for (size_t i = 1; i < numOfHidden.size(); i++)
        {
            weights.at(i) =
                Matrix<double>(numOfHidden.at(i), numOfHidden.at(i - 1));
            biases.at(i) = Matrix<double>::RowVector(numOfHidden.at(i));
            deltas.at(i) = Matrix<double>::RowVector(numOfHidden.at(i));
        }
        weights.at(numOfHidden.size()) =
            Matrix<double>(numOfOutput, Last(numOfHidden));
        biases.at(numOfHidden.size()) = Matrix<double>::RowVector(numOfOutput);
        deltas.at(numOfHidden.size()) = Matrix<double>::RowVector(numOfOutput);

        for (size_t i = 0; i < numOfHidden.size(); i++)
        {
            hidden.at(i) = Matrix<double>::RowVector(numOfHidden.at(i));
        }
        for (Matrix<double>& item : weights)
        {
            RandomizeMatrix(item);
        }
        for (Matrix<double>& item : biases)
        {
            RandomizeMatrix(item);
        }
    }
}

void BPNeuralNetwork::FeedForward()
{
    Activte(First(hidden), First(weights) * input + First(biases));
    for (size_t i = 1; i < hidden.size(); i++)
    {
        Activte(hidden.at(i), weights.at(i) * hidden.at(i - 1) + biases.at(i));
    }
    Activte(output, Last(weights) * Last(hidden) + Last(biases));
}

void BPNeuralNetwork::Backpropagation(const Matrix<double>& targets)
{
    BackpropagationToOutputLayer(targets);
    BackpropagationToHiddenLayers();
    UpdateWeights();
}

void BPNeuralNetwork::BackpropagationToOutputLayer(
    const Matrix<double>& targets)
{
    Matrix<double> errorFactor =
        targets.Apply(output, [](double x, double y) { return -(x - y); });

    Matrix<double> derivative = DerivativeActive(output);
    int length = derivative.Rows() * derivative.Cols();
    for (int i = 0; i < length; i++)
    {
        Last(deltas)(i) = derivative(i) * errorFactor(i);
    }
}

void BPNeuralNetwork::BackpropagationToHiddenLayers()
{
    for (int k = static_cast<int>(hidden.size() - 1); k >= 0; k--)
    {
        size_t uk = static_cast<size_t>(k);
        Matrix<double> errorFactor =
            (deltas.at(uk + 1).Transpose() * weights.at(uk + 1)).Transpose();
        Matrix<double> derivative = DerivativeActive(hidden.at(uk));
        int length = derivative.Rows() * derivative.Cols();
        for (int i = 0; i < length; i++)
        {
            deltas.at(uk)(i) = derivative(i) * errorFactor(i);
        }
    }
}

void BPNeuralNetwork::UpdateWeights()
{
    UpdateFirstWeights();
    for (int k = static_cast<int>(hidden.size() - 2); k >= 0; k--)
    {
        size_t uk = static_cast<size_t>(k);
        Matrix<double> newBias =
            biases.at(uk + 1) - learningRate * deltas.at(uk + 1);
        int length = newBias.Rows() * newBias.Cols();
        for (int i = 0; i < length; i++)
        {
            biases.at(uk + 1)(i) = newBias(i);
        }
        Matrix<double>& weight = weights.at(uk + 1);

        for (int i = 0; i < weight.Rows(); i++)
        {
            for (int j = 0; j < weight.Cols(); j++)
            {
                weight(i, j) = weight(i, j) - learningRate *
                                                  hidden.at(uk)(0, j) *
                                                  deltas.at(uk + 1)(i);
            }
        }
    }
    UpdateLastWeights();
}

void BPNeuralNetwork::UpdateLastWeights()
{

    Matrix<double> newBias = Last(biases) - learningRate * Last(deltas);
    int length = newBias.Rows() * newBias.Cols();
    for (int i = 0; i < length; i++)
    {
        Last(biases)(i) = newBias(i);
    }
    Matrix<double>& weight = Last(weights);
    for (int i = 0; i < weight.Rows(); i++)
    {
        for (int j = 0; j < weight.Cols(); j++)
        {
            weight(i, j) = weight(i, j) -
                           learningRate * Last(hidden)(0, j) * Last(deltas)(i);
        }
    }
}

void BPNeuralNetwork::UpdateFirstWeights()
{
    Matrix<double> newBias = First(biases) - learningRate * First(deltas);
    int length = newBias.Rows() * newBias.Cols();
    for (int i = 0; i < length; i++)
    {
        First(biases)(i) = newBias(i);
    }
    Matrix<double>& weight = First(weights);
    for (int i = 0; i < weight.Rows(); i++)
    {
        for (int j = 0; j < weight.Cols(); j++)
        {
            weight(i, j) =
                weight(i, j) - learningRate * input(0, j) * First(deltas)(i);
        }
    }
}

void BPNeuralNetwork::RandomizeMatrix(Matrix<double>& matrix)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);
    int length = matrix.Rows() * matrix.Cols();
    for (int i = 0; i < length; i++)
    {
        matrix(i) = distribution(generator);
    }
}

void Activte(Matrix<double>& result, const Matrix<double>& matrix)
{
    for (int i = 0; i < matrix.Rows() * matrix.Cols(); i++)
    {
        result(i) = 1.0 / (1.0 + std::exp(-matrix(i)));
    }
}

Matrix<double> DerivativeActive(const Matrix<double>& matrix)
{
    return matrix.Apply([](double x) { return x * (1.0 - x); });
}
