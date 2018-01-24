#include "BPNeutralNetwork.h"
#include <cmath>
#include <iostream>
using namespace std;

double Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double DerivedSigmoid(double x)
{
    return x * (1 - x);
}
double Square(double x)
{
    return x * x;
}

BPNeutralNetwork::BPNeutralNetwork(size_t numOfInputs,
                                   vector<size_t> numOfHidden,
                                   size_t numOfOutputs)
    : inputLayer(numOfInputs), hiddenLayers(numOfHidden.size()),
      outputLayer(numOfOutputs), weights(1 + numOfHidden.size()),
      hiddenErrors(numOfHidden.size()), outputErrors(numOfOutputs)
{
    if (numOfHidden.size() < 1)
    {
        throw "The number of hidden layers should be greater than one.";
    }
    else
    {

        weights.at(0) = Matrix(inputLayer.TotalSize(), numOfHidden.at(0));
        for (size_t i = 0; i < numOfHidden.size(); i++)
        {
            hiddenLayers.at(i) = Layer(numOfHidden.at(i));
        }
        for (size_t i = 1; i < weights.size() - 1; i++)
        {
            size_t nRows = hiddenLayers.at(i - 1).ValuesSize();
            size_t nCols = numOfHidden.at(i);
            weights.at(i) = Matrix(nRows, nCols);
        }
        Layer& lastHiddenLayer = hiddenLayers.at(hiddenLayers.size() - 1);

        weights.at(weights.size() - 1) =
            Matrix(lastHiddenLayer.ValuesSize(), outputLayer.ValuesSize());

        for (size_t i = 1; i < weights.size(); i++)
        {
            Matrix& matrix = weights.at(i);
            hiddenErrors.at(i - 1) =
                Matrix(matrix.RowCount(), matrix.ColumnCount());
        }
        RandomizeWeights();
    }
}

void BPNeutralNetwork::SetInputLayer(NumericVector input, double intercept)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        inputLayer.At(i) = input.at(i);
    }
    inputLayer.intercept = intercept;
}

void BPNeutralNetwork::SetHiddenLayer(size_t k, NumericVector hidden,
                                      double intercept)
{
    for (size_t i = 0; i < hidden.size(); i++)
    {
        hiddenLayers.at(k).At(i) = hidden.at(i);
    }
    hiddenLayers.at(k).intercept = intercept;
}

void BPNeutralNetwork::SetWeight(size_t i, Matrix matrix)
{
    weights.at(i) = matrix;
}

void BPNeutralNetwork::FeedForward()
{
    FeedForwardInput();
    FeedForwardHidden();
    FeedForwardOutput();
}

void BPNeutralNetwork::FeedForwardInput()
{
    Matrix& matrix = weights.at(0);
    Layer& firstHiddenLayer = hiddenLayers.at(0);
    for (size_t j = 0; j < firstHiddenLayer.ValuesSize(); j++)
    {
        double sum = 0.0;
        for (size_t i = 0; i < inputLayer.ValuesSize(); i++)
        {
            sum = sum + inputLayer.At(i) * matrix(i, j);
        }
        firstHiddenLayer.At(j) = Sigmoid(sum + inputLayer.intercept);
    }
}

void BPNeutralNetwork::FeedForwardHidden()
{
    for (size_t k = 0; k < hiddenLayers.size() - 1; k++)
    {
        Matrix& matrix = weights.at(k + 1);
        for (size_t j = 0; j < hiddenLayers.at(k + 1).ValuesSize(); j++)
        {
            double sum = 0.0;
            for (size_t i = 0; i < hiddenLayers.at(k).ValuesSize(); i++)
            {
                sum = sum + hiddenLayers.at(k).At(i) * matrix(i, j);
            }
            hiddenLayers.at(k + 1).At(j) =
                Sigmoid(sum + hiddenLayers.at(k).intercept);
        }
    }
}

void BPNeutralNetwork::FeedForwardOutput()
{
    Matrix& matrix = weights.at(weights.size() - 1);
    for (size_t j = 0; j < outputLayer.ValuesSize(); j++)
    {
        double sum = 0.0;
        Layer& lastHiddenLayer = hiddenLayers.at(hiddenLayers.size() - 1);
        for (size_t i = 0; i < lastHiddenLayer.ValuesSize(); i++)
        {
            sum = sum + lastHiddenLayer.At(i) * matrix(i, j);
        }
        outputLayer.At(j) = Sigmoid(sum + lastHiddenLayer.intercept);
    }
}

void BPNeutralNetwork::ComputeErrors(NumericVector target)
{
    ComputeOutputErrors(target);
    ComputeHiddenErrors();
    ComputeInputErrors();
}

void BPNeutralNetwork::ComputeOutputErrors(NumericVector& target)
{
    //    cout << "computing output errors" << endl;
    Matrix& lastHiddenErrors = hiddenErrors.at(hiddenErrors.size() - 1);
    Layer& lastHiddenLayer = hiddenLayers.at(hiddenLayers.size() - 1);
    Matrix& lastWeights = weights.at(weights.size() - 1);

    for (size_t j = 0; j < outputLayer.ValuesSize(); j++)
    {
        double out = outputLayer.At(j);
        double total_error_out = -(target.at(j) - out);
        double out_net = DerivedSigmoid(out);

        //        cout << "Total Error: " << total_error_out << endl;
        outputErrors.at(j) = total_error_out;
        for (size_t i = 0; i < lastHiddenErrors.RowCount(); i++)
        {
            double net_w = lastHiddenLayer.At(i);
            double total_error_w = total_error_out * out_net * net_w;
            double old_weight = lastWeights(i, j);
            lastWeights(i, j) = old_weight - learningRate * total_error_w;
            lastHiddenErrors(i, j) = total_error_out * out_net * old_weight;
        }
    }
}

void BPNeutralNetwork::ComputeHiddenErrors()
{
    //    cout << "computing hidden errors" << endl;

    for (size_t k = hiddenLayers.size() - 1; k > 0; k--)
    {
        Matrix& nextHiddenErrors = hiddenErrors.at(k);
        Layer& nextHiddenLayer = hiddenLayers.at(k);
        Matrix& previousHiddenWeights = weights.at(k - 1);
        Layer& previousHiddenLayer = hiddenLayers.at(k - 1);

        for (size_t j = 0; j < nextHiddenLayer.ValuesSize(); j++)
        {
            for (size_t i = 0; i < previousHiddenLayer.ValuesSize(); i++)
            {
                double total_error_out = nextHiddenErrors.SumOfRow(i);
                double out_net = DerivedSigmoid(nextHiddenLayer.At(j));
                double total_error_weight =
                    total_error_out * out_net * previousHiddenLayer.At(i);
                double old_weight = previousHiddenWeights(i, j);
                previousHiddenWeights(i, j) =
                    old_weight - learningRate * total_error_weight;
            }
        }
    }
}

void BPNeutralNetwork::ComputeInputErrors()
{
    //    cout << "computing input errors" << endl;
    Matrix& firstHiddenErrors = hiddenErrors.at(0);

    Layer& firstHiddenLayer = hiddenLayers.at(0);
    Matrix& inputWeights = weights.at(0);

    for (size_t j = 0; j < firstHiddenErrors.RowCount(); j++)
    {
        for (size_t i = 0; i < inputLayer.ValuesSize(); i++)
        {
            double total_error_out = firstHiddenErrors.SumOfRow(i);
            double out_net = DerivedSigmoid(firstHiddenLayer.At(j));
            double total_error_weight =
                total_error_out * out_net * inputLayer.At(i);
            double old_weight = inputWeights(i, j);
            inputWeights(i, j) = old_weight - learningRate * total_error_weight;
        }
    }
}

void BPNeutralNetwork::Display()
{
    cout << "************* start displaying *************" << endl;
    size_t i = 0;
    for (Matrix& item : weights)
    {
        cout << "Weight " << i << ": " << endl;
        i++;
        item.Display();
    }
    cout << endl << "input layer: " << endl;
    cout << inputLayer;

    cout << endl << "hidden layer: " << endl;
    for (Layer& item : hiddenLayers)
    {
        cout << item;
    }
    cout << endl << "output layer: " << endl;
    cout << outputLayer;

    cout << "************* end displaying *************" << endl;
}

void BPNeutralNetwork::RandomizeWeights()
{
    for (Matrix& item : weights)
    {
        item.RandomSet();
    }
}

double BPNeutralNetwork::TotalErrorSquare()
{
    double sum = 0.0;
    for (double item : outputErrors)
    {
        sum += Square(item);
    }
    return sum;
}
