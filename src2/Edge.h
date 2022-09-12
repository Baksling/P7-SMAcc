#ifndef EDGE_H
#define EDGE_H

#include "Node.h"

class edge {
    private:
        node* n1_;
        node* n2_;
        float weight_;
    public:
        edge(node* n1, node* n2, float weight);
        float get_weight() const;
        node* get_n1() const;
        node* get_n2() const;
};



#endif // EDGE_H