#include "Test.hpp"
#include "BPNeuralNetwork.hpp"
#include <iostream>
using namespace std;
void TestFeedForward()
{
    BPNeuralNetwork network(2, {2}, 2);
    network.input = Matrix<double>(2, 1, {0.05, 0.1});
    network.weights.at(0) = Matrix<double>(2, 2, {0.15, 0.20, 0.25, 0.30});
    network.weights.at(1) = Matrix<double>(2, 2, {0.40, 0.45, 0.50, 0.55});
    network.biases.at(0) = Matrix<double>(2, 1, {0.35, 0.35});
    network.biases.at(1) = Matrix<double>(2, 1, {0.60, 0.60});
    Matrix<double> targets(2, 1, {0.01, 0.99});
    for (int i = 0; i < 50000; i++)
    {
        network.FeedForward();
        network.BackpropagationToOutputLayer(targets);
        if (i % 10000 == 0)
        {
            cout << "output: " << endl << network.output << endl;
        }
    }
    //    cout << "hidden weights:" << endl << network.weights.at(1) << endl;
}

void Test1()
{
    BPNeuralNetwork network(3, {3, 3}, 1);
    vector<Matrix<double>> X;
    X.push_back(Matrix<double>(3, 1, {0, 0, 1}));
    X.push_back(Matrix<double>(3, 1, {0, 1, 1}));
    X.push_back(Matrix<double>(3, 1, {1, 0, 1}));
    X.push_back(Matrix<double>(3, 1, {1, 1, 1}));

    vector<double> Y = {0, 1, 1, 0};
    for (int i = 0; 10000; i++)
    {
        for (size_t j = 0; j < X.size(); j++)
        {
            network.input = X.at(j);
            Matrix<double> targets = Matrix<double>(1, 1, {Y.at(j)});
            network.FeedForward();
            network.BackpropagationToOutputLayer(targets);
            if (i % 10000 == 0)
            {
                cout << "output: " << endl;
                for (size_t k = 0; k < Y.size(); k++)
                {
                    network.input = X.at(k);
                    network.FeedForward();
                    cout << network.output << endl;
                }
            }
        }
    }
}
