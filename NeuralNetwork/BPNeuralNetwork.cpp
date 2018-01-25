#include "BPNeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <stdlib.h>
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

BPNeuralNetwork::BPNeuralNetwork(size_t numOfInputs, vector<size_t> numOfHidden,
                                 size_t numOfOutputs)
    : inputLayer(numOfInputs), hiddenLayers(numOfHidden.size()),
      outputLayer(numOfOutputs), weights(1 + numOfHidden.size()),
      hiddenErrors(numOfHidden.size()), outputErrors(numOfOutputs),
      hiddenBiasErrors(numOfHidden.size())
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
            hiddenBiasErrors.at(i - 1) = NumericVector(matrix.ColumnCount());
        }
        RandomizeWeights();
    }
}

void BPNeuralNetwork::SetInputLayer(NumericVector input)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        inputLayer.At(i) = input.at(i);
    }
}

void BPNeuralNetwork::SetInputLayer(NumericVector input, double intercept)
{
    for (size_t i = 0; i < input.size(); i++)
    {
        inputLayer.At(i) = input.at(i);
    }
    inputLayer.intercept = intercept;
}

void BPNeuralNetwork::SetHiddenLayer(size_t k, NumericVector hidden,
                                     double intercept)
{
    for (size_t i = 0; i < hidden.size(); i++)
    {
        hiddenLayers.at(k).At(i) = hidden.at(i);
    }
    hiddenLayers.at(k).intercept = intercept;
}

void BPNeuralNetwork::SetWeight(size_t i, Matrix matrix)
{
    weights.at(i) = matrix;
}

void BPNeuralNetwork::FeedForward()
{
    FeedForwardInput();
    FeedForwardHidden();
    FeedForwardOutput();
}

void BPNeuralNetwork::FeedForwardInput()
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

void BPNeuralNetwork::FeedForwardHidden()
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

void BPNeuralNetwork::FeedForwardOutput()
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

void BPNeuralNetwork::ComputeErrors(NumericVector target)
{
    ComputeOutputErrors(target);
    ComputeHiddenErrors();
    ComputeInputErrors();
}

void BPNeuralNetwork::ComputeOutputErrors(NumericVector& target)
{
    //    cout << "computing output errors" << endl;
    Matrix& lastHiddenErrors = hiddenErrors.at(hiddenErrors.size() - 1);
    Layer& lastHiddenLayer = hiddenLayers.at(hiddenLayers.size() - 1);
    Matrix& lastWeights = weights.at(weights.size() - 1);
    NumericVector& lastHiddenBiasErrors =
        hiddenBiasErrors.at(hiddenBiasErrors.size() - 1);
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
        /* adjust bias */
        double old_intercept = lastHiddenLayer.intercept;
        double total_error_intercept =
            total_error_out * out_net * old_intercept;
        lastHiddenLayer.intercept =
            old_intercept - learningRate * total_error_intercept;
        lastHiddenBiasErrors.at(j) = total_error_out * out_net * old_intercept;
    }
}

void BPNeuralNetwork::ComputeHiddenErrors()
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
            double out_net = DerivedSigmoid(nextHiddenLayer.At(j));
            for (size_t i = 0; i < previousHiddenLayer.ValuesSize(); i++)
            {
                double total_error_out = nextHiddenErrors.SumOfRow(i);

                double total_error_weight =
                    total_error_out * out_net * previousHiddenLayer.At(i);
                double old_weight = previousHiddenWeights(i, j);
                previousHiddenWeights(i, j) =
                    old_weight - learningRate * total_error_weight;
            }

            /* adjust bias */
            {
                double total_error_out =
                    SumNumericVector(hiddenBiasErrors.at(k));
                double old_intercept = previousHiddenLayer.intercept;
                double total_error_weight =
                    total_error_out * out_net * old_intercept;
                hiddenBiasErrors.at(k - 1).at(j) =
                    old_intercept - learningRate * total_error_weight;
            }
        }
    }
}

void BPNeuralNetwork::ComputeInputErrors()
{
    //    cout << "computing input errors" << endl;
    Matrix& firstHiddenErrors = hiddenErrors.at(0);

    Layer& firstHiddenLayer = hiddenLayers.at(0);
    Matrix& inputWeights = weights.at(0);
    NumericVector& firstHiddenBiasErrors = hiddenBiasErrors.at(0);

    for (size_t j = 0; j < firstHiddenErrors.RowCount(); j++)
    {
        double out_net = DerivedSigmoid(firstHiddenLayer.At(j));
        for (size_t i = 0; i < inputLayer.ValuesSize(); i++)
        {
            double total_error_out = firstHiddenErrors.SumOfRow(i);
            double total_error_weight =
                total_error_out * out_net * inputLayer.At(i);
            double old_weight = inputWeights(i, j);
            inputWeights(i, j) = old_weight - learningRate * total_error_weight;
        }

        /* adjust bias */
        {
            double old_intercept = inputLayer.intercept;
            double total_error_out = SumNumericVector(firstHiddenBiasErrors);
            double total_error_weight =
                total_error_out * out_net * old_intercept;
            inputLayer.intercept =
                old_intercept - learningRate * total_error_weight;
        }
    }
}

void BPNeuralNetwork::Display()
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

void BPNeuralNetwork::RandomizeWeights()
{
    for (Matrix& item : weights)
    {
        item.RandomSet();
    }
    inputLayer.intercept = rand() / double(RAND_MAX);
    for (Layer& layer : hiddenLayers)
    {
        layer.intercept = rand() / double(RAND_MAX);
    }
}

double BPNeuralNetwork::TotalErrorSquare()
{
    double sum = 0.0;
    for (double item : outputErrors)
    {
        sum += Square(item);
    }
    return sum;
}

double SumNumericVector(vector<double>& vec)
{
    double sum = 0.0;
    for (double& item : vec)
    {
        sum = sum + item;
    }
    return sum;
}

double MaxNumericVector(vector<double>& vec)
{
    double max = vec.at(0);
    for (size_t i = 1; i < vec.size(); i++)
    {
        double& item = vec.at(i);
        if (max < item)
        {
            max = item;
        }
    }
    return max;
}

vector<double> Standardize(vector<double>& input)
{
    double average = AverageNumericVector(input);
    double max = MaxNumericVector(input);
    cout << "average = " << average << endl;
    cout << "max = " << max << endl;
    vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        output.at(i) = (input.at(i) - average) / max;
    }
    return output;
}

double AverageNumericVector(vector<double>& vec)
{
    double sum = SumNumericVector(vec);
    return sum / double(vec.size());
}

void DisplayNumericVector(vector<double>& input)
{
    cout << "vector(" << input.size() << "): " << endl;
    for (double& item : input)
    {
        cout << item << "  ";
    }
    cout << endl;
}
