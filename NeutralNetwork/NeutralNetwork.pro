TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    BPNeutralNetwork.cpp \
    Matrix.cpp \
    Layer.cpp \
    Test.cpp

HEADERS += \
    BPNeutralNetwork.h \
    Matrix.h \
    Layer.h \
    Test.h
