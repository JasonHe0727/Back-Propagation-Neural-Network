TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    BPNeuralNetwork.cpp \
    Matrix.cpp \
    Layer.cpp \
    Test.cpp

HEADERS += \
    BPNeuralNetwork.h \
    Matrix.h \
    Layer.h \
    Test.h
