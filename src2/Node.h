#ifndef NODE_H
#define NODE_H

#include <list>
using namespace std;  // NOLINT(clang-diagnostic-header-hygiene)

class node {
    public:
        int id;
        int get_id() const;
        explicit node(int id);
};

#endif // NODE_H