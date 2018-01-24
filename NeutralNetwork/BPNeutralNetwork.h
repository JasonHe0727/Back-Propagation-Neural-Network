#ifndef BPNEUTRALNETWORK_H
#define BPNEUTRALNETWORK_H

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

class BPNeutralNetwork
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
    BPNeutralNetwork(size_t numOfInputs, vector<size_t> numOfHidden,
                     size_t numOfOutputs);
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

#endif // BPNEUTRALNETWORK_H
