#include "cuda_allocator.h"

automata* cuda_allocator::allocate_automata(const automata* source)
{
    const size_t network_size = sizeof(void*)*source->network.size;
    node* node_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&node_store, sizeof(node)*source->network.size));
    node** local_network = static_cast<node**>(malloc(network_size));
    
    for (int i = 0; i < source->network.size; ++i)
    {
        this->allocate_node(source->network.store[i], &node_store[i]);
        local_network[i] = &node_store[i];
    }

    node** network = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&network, network_size));
    CUDA_CHECK(cudaMemcpy(network, local_network, network_size, cudaMemcpyHostToDevice));

    clock_var* var_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&var_store, sizeof(clock_var)*source->variables.size));
    
    for (int i = 0; i < source->variables.size; ++i)
    {
        allocate_clock(&source->variables.store[i], &var_store[i]);
    }

    const automata temp = {
        arr<node*>{ network, source->network.size },
        arr<clock_var>{ var_store, source->variables.size },
    };

    automata* dest = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&dest, sizeof(automata)));
    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(automata), cudaMemcpyHostToDevice));

    return dest;
}

void cuda_allocator::allocate_node(const node* source, node* dest)
{
    if(this->circular_ref_.count(source)) return;
    this->circular_ref_.insert(std::pair<const node*, node*>(source, dest));
    
    edge* edge_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&edge_store, sizeof(edge)*source->edges.size));
    for (int i = 0; i < source->edges.size; ++i)
    {
        allocate_edge(&source->edges.store[i], &edge_store[i]);
    }

    constraint* invariant_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&invariant_store, sizeof(constraint)*source->invariants.size));
    for (int i = 0; i < source->invariants.size; ++i)
    {
        allocate_constraint(&source->invariants.store[i], &invariant_store[i]);
    }

    expr* exp_d = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)));
    this->allocate_expr(source->lamda, exp_d);

    const node temp = node{
        source->id,
        exp_d,
        arr<edge>{edge_store, source->edges.size},
        arr<constraint>{invariant_store, source->invariants.size},
        source->is_branch_point,
        source->is_goal
    };

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(node), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_edge(const edge* source, edge* dest)
{
    //Handle node circular reference.
    node* node_dest = nullptr;
    if(this->circular_ref_.count(source->dest) == 0)
    {
        CUDA_CHECK(this->allocator_->allocate(&node_dest, sizeof(node)));
        this->allocate_node(source->dest, node_dest);
    }
    else
    {
        node_dest = this->circular_ref_[source->dest];
    }

    constraint* guard_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&guard_store, sizeof(constraint)*source->guards.size));
    for (int i = 0; i < source->guards.size; ++i)
    {
        allocate_constraint(&source->guards.store[i], &guard_store[i]);
    }

    update* update_store = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&update_store, sizeof(update)*source->updates.size));
    for (int i = 0; i < source->updates.size; ++i)
    {
        allocate_update(&source->updates.store[i], &update_store[i]);
    }

    expr* exp_d = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&exp_d, sizeof(expr)));
    this->allocate_expr(source->weight, exp_d);

    const edge temp = edge{
        source->channel,
        exp_d,
        node_dest,
        arr<constraint>{ guard_store, source->guards.size },
        arr<update>{ update_store, source->updates.size }
    };

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(edge), cudaMemcpyHostToDevice));

}

void cuda_allocator::allocate_constraint(const constraint* source, constraint* dest)
{
    constraint temp{};
    temp.operand = source->operand;
    temp.uses_variable = source->uses_variable;

    if(source->uses_variable)
    {
        temp.variable_id = source->variable_id;
    }
    else
    {
        expr* left = nullptr;
        CUDA_CHECK(this->allocator_->allocate(&left, sizeof(expr)));
        this->allocate_expr(source->value, left);
        temp.value = left;
    }

    expr* right = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
    this->allocate_expr(source->expression, right);
    temp.expression = right;

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(constraint), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_update(const update* source, update* dest)
{
    expr* right = nullptr;
    CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
    this->allocate_expr(source->expression, right);

    const update temp{
        source->variable_id,
        right
    };
    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(update), cudaMemcpyHostToHost));
}

// ReSharper disable once CppMemberFunctionMayBeStatic
void cuda_allocator::allocate_clock(const clock_var* source, clock_var* dest) const
{
    CUDA_CHECK(cudaMemcpy(dest, source, sizeof(clock_var), cudaMemcpyHostToDevice));
}

void cuda_allocator::allocate_expr(const expr* source, expr* dest)
{
    if(IS_LEAF(source->operand))
    {
        CUDA_CHECK(cudaMemcpy(dest, source, sizeof(expr), cudaMemcpyHostToDevice));
        return;
    }

    expr* left = nullptr;
    if(source->left != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&left, sizeof(expr)));
        this->allocate_expr(source->left, left);
    }

    expr* right = nullptr;
    if(source->right != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&right, sizeof(expr)));
        this->allocate_expr(source->right, right);
    }

    expr* else_branch = nullptr;
    if(source->operand == expr::conditional_ee && source->conditional_else != nullptr)
    {
        CUDA_CHECK(this->allocator_->allocate(&else_branch, sizeof(expr)));
        this->allocate_expr(source->conditional_else, else_branch);
    }

    expr temp{};
    temp.left = left;
    temp.right = right;
    temp.operand = source->operand;

    //this shouldn't be necessary, but safety is nr. 1 priority.
    if     (source->operand == expr::literal_ee)        temp.value            = source->value;
    else if(source->operand == expr::clock_variable_ee) temp.variable_id      = source->variable_id;
    else if(source->operand == expr::conditional_ee)    temp.conditional_else = else_branch;

    CUDA_CHECK(cudaMemcpy(dest, &temp, sizeof(expr), cudaMemcpyHostToDevice));
}
