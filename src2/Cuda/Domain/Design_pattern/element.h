#pragma once

#ifndef ELEMENT_H
#define ELEMENT_H

#include "visistor.h"

class element
{
protected:
    ~element() = default;
public:
    virtual void accept(visistor& v) = 0;
};

#endif