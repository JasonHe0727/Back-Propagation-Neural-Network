#include "Test.h"
#include "BPNeutralNetwork.h"
#include <Layer.h>
#include <cmath>
#include <iostream>
using namespace std;
void TestIterator()
{
    Layer layer({1, 2, 3}, 43);
    for (double item : layer)
    {
        cout << item << endl;
    }
}
void TestNN()
{
    cout << "Let's rock and roll!" << endl;
    BPNeutralNetwork network(2, {2}, 2);
    network.SetInputLayer({0.05, 0.10}, 0.35);
    network.SetHiddenLayer(0, {0, 0}, 0.60);
    Matrix w1(2, 2, {0.15, 0.25, 0.20, 0.30});
    Matrix w2(2, 2, {0.40, 0.50, 0.45, 0.55});
    network.SetWeight(0, w1);
    network.SetWeight(1, w2);

    network.FeedForward();
    network.Display();

    NumericVector vec{0.01, 0.99};
    network.ComputeErrors(vec);
    network.Display();
}

void Test1()
{
    auto sigmoid = [](double x) { return 1.0 / (1.0 + exp(-x)); };
    vector<double> x1 = {-3,   -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6,
                         -0.3, 0,	0.3,  0.6,  0.9,  1.2,  1.5,  1.8};
    vector<double> x2 = {-2,   -1.8,	-1.6, -1.4, -1.2, -1,  -0.8, -0.6, -0.4,
                         -0.2, -2.2204, 0.2,  0.4,  0.6,  0.8, 1,	1.2};
    vector<double> y = {0.6589,  0.2206,  -0.1635, -0.4712, -0.6858,
                        -0.7975, -0.8040,

                        -0.7113, -0.5326, -0.2875, 0,		0.3035,
                        0.5966,  0.8553,  1.0600,  1.1975,  1.2618};
    BPNeutralNetwork network(2, {3, 3, 3}, 1);
    network.learningRate = 0.5;
    vector<double> y_predict(y.size());
    for (int count = 0; count < 10000; count++)
    {
        double sum = 0.0;

        for (size_t i = 0; i < x1.size(); i++)
        {
            network.SetInputLayer({x1.at(i), x2.at(i)}, 0.35);
            network.FeedForward();
            network.ComputeErrors({sigmoid(y.at(i))});
            sum += network.TotalErrorSquare();
            y_predict.at(i) = network.outputLayer.At(0);
        }
        cout << "id = " << (count + 1) << "  total error: " << sum << endl;
    }
    cout << "output: " << endl;
    for (double item : y_predict)
    {
        cout << item << ", ";
    }
    cout << endl;
}

void Test2()
{
    cout << "Let's rock and roll!" << endl;
    BPNeutralNetwork network(2, {2}, 2);
    network.SetInputLayer({0.05, 0.10}, 0.35);
    network.SetHiddenLayer(0, {0, 0}, 0.60);
    Matrix w1(2, 2, {0.15, 0.25, 0.20, 0.30});
    Matrix w2(2, 2, {0.40, 0.50, 0.45, 0.55});
    network.SetWeight(0, w1);
    network.SetWeight(1, w2);

    network.FeedForward();

    for (int count = 0; count < 10000; count++)
    {
        network.FeedForward();
        network.ComputeErrors({0.01, 0.99});
        cout << "id = " << (count + 1)
             << "  total error: " << network.TotalErrorSquare() << endl;
    }
    network.Display();
}
double Average(vector<double>& vec)
{
    double sum = 0.0;
    for (double item : vec)
    {
        sum = sum + item;
    }
    return sum / static_cast<double>(vec.size());
}
vector<double> Subtrac(vector<double>& x, vector<double> y)
{
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        result.at(i) = x.at(i) - y.at(i);
    }
    return result;
}
void Test3()
{
    vector<double> x1 = {-0.71428571, -0.71428571, -0.14285714,
                         0.42857143,  1,		   0.14285714};
    vector<double> x2 = {-0.36363636, 0.31818182, 1,
                         -0.77272727, -0.5,		  0.31818182};

    vector<double> y = {1, 1, 1, 0, 0, 0};
    BPNeutralNetwork network(2, {3, 3, 3, 3}, 1);
    network.learningRate = 0.5;
    vector<double> y_predict(y.size());
    for (int count = 0; count < 1000000; count++)
    {
        double sum = 0.0;

        for (size_t i = 0; i < x1.size(); i++)
        {
            network.SetInputLayer({x1.at(i), x2.at(i)}, 0.35);
            network.FeedForward();
            network.ComputeErrors({y.at(i)});
            sum += network.TotalErrorSquare();
            y_predict.at(i) = network.outputLayer.At(0);
        }
        cout << "id = " << (count + 1) << "  total error: " << sum << endl;
    }
    cout << "output: " << endl;
    for (double item : y_predict)
    {
        cout << item << ", ";
    }
    cout << endl;
}
