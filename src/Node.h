#include <list>

using namespace std;

#ifndef NODE_H
#define NODE_H

class Node {
    public:
        int id;
        int GetId();
        Node(int _id);
};

#endif // NODE_H