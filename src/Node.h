#import <list>

#ifndef NODE_H
#define NODE_H

#define OUTPUT_BUFFER 10

class Node {
    public:
        int id;
        int GetId();
        Node(int _id);
        list<Node> outputs(OUTPUT_BUFFER);
};

#endif // NODE_H