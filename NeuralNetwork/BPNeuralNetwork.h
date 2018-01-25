#ifndef BPNEURALNETWORK_H
#define BPNEURALNETWORK_H

#include "Layer.h"
#include "Matrix.h"
#include <memory>
#include <vector>

using std::vector;
using NumericVector = vector<double>;
using LayerList = vector<Layer>;
using MatrixList = vector<Matrix>;
double Sigmoid(double x);
double DerivedSigmoid(double x);
double Square(double x);

class BPNeuralNetwork
{
public:
    double learningRate = 0.5;

    Layer inputLayer;
    LayerList hiddenLayers;
    Layer outputLayer;
    MatrixList weights;

    NumericVector inputErrors;
    MatrixList hiddenErrors;
    NumericVector outputErrors;
    vector<NumericVector> hiddenBiasErrors;

    BPNeuralNetwork(size_t numOfInputs, vector<size_t> numOfHidden,
                    size_t numOfOutputs);
    void SetInputLayer(NumericVector input);
    void SetInputLayer(NumericVector input, double intercept);
    void SetHiddenLayer(size_t k, NumericVector hidden, double intercept);
    void SetWeight(size_t i, Matrix matrix);

    void FeedForward();

    void ComputeErrors(NumericVector target);

    void Display();
    void RandomizeWeights();
    double TotalErrorSquare();

private:
    void FeedForwardInput();
    void FeedForwardHidden();
    void FeedForwardOutput();

    void ComputeOutputErrors(NumericVector& target);
    void ComputeHiddenErrors();
    void ComputeInputErrors();
};

double SumNumericVector(vector<double>& vec);
double MaxNumericVector(vector<double>& vec);
double AverageNumericVector(vector<double>& vec);
vector<double> Standardize(vector<double>& input);
void DisplayNumericVector(vector<double>& input);
#endif // BPNEURALNETWORK_H
