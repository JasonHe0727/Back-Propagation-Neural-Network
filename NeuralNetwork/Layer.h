#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
using std::ostream;
using std::vector;
class Layer;
class LayerIterator;

class Layer
{
public:
    vector<double> values;
    double intercept;
    Layer();
    Layer(size_t number);
    Layer(vector<double> initializers, double intercept);
    double& At(size_t i);
    LayerIterator begin();
    LayerIterator end();
    size_t TotalSize();
    size_t ValuesSize();
    friend ostream& operator<<(ostream& out, Layer& layer);
};

class LayerIterator
{
public:
    Layer& layer;
    size_t current;
    LayerIterator(Layer& layer, size_t current);
    LayerIterator& operator++();
    bool operator!=(LayerIterator& other);
    double operator*();
};
#endif // LAYER_H
