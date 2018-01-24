#include "Layer.h"
using namespace std;
Layer::Layer() : intercept(0)
{
}

Layer::Layer(size_t number) : values(number), intercept(0)
{
}

Layer::Layer(vector<double> initializers, double intercept)
    : values(initializers), intercept{intercept}
{
}

double& Layer::At(size_t i)
{
    return values.at(static_cast<size_t>(i));
}

LayerIterator Layer::begin()
{
    return LayerIterator(*this, 0);
}

LayerIterator Layer::end()
{
    return LayerIterator(*this, values.size());
}

size_t Layer::TotalSize()
{
    return values.size() + 1;
}

size_t Layer::ValuesSize()
{
    return values.size();
}

std::ostream& operator<<(std::ostream& out, Layer& layer)
{
    out << "layer: " << endl;
    for (double item : layer.values)
    {
        out << item << "  ";
    }
    out << endl;
    out << "intercept: " << layer.intercept << endl;
    return out;
}

LayerIterator::LayerIterator(Layer& layer, size_t current)
    : layer{layer}, current{current}
{
}

LayerIterator& LayerIterator::operator++()
{
    current++;
    return *this;
}

bool LayerIterator::operator!=(LayerIterator& other)
{
    return current != other.current;
}

double LayerIterator::operator*()
{
    return layer.values.at(current);
}
