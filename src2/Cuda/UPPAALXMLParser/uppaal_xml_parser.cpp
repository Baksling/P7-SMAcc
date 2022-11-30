﻿#include "uppaal_xml_parser.h"


/* 
 * TODO init like this: double x,y = 0.0; --flueben
 * TODO If else in init, guards, and invariants
 * TODO Clean up, e.g. add trimmer for whitespace AND TABS! --flueben
 */

constraint* get_constraint(const string& exprs, const int timer_id, expr* value)
{
    constraint* cons = new constraint();
    if(exprs.find("<=") != std::string::npos)
        cons->operand = constraint::less_equal_c;
    else if(exprs.find(">=") != std::string::npos)
        cons->operand = constraint::greater_equal_c;
    else if(exprs.find("==") != std::string::npos)
        cons->operand = constraint::equal_c;
    else if(exprs.find("!=") != std::string::npos)
        cons->operand = constraint::not_equal_c;
    else if(exprs.find('<') != std::string::npos)
        cons->operand = constraint::less_c;
    else if(exprs.find('>') != std::string::npos)
        cons->operand = constraint::greater_c;
    else
    {
        THROW_LINE("Operand in " + exprs + " not found, sad..");
    }

    cons->uses_variable = true;
    cons->expression = value;
    cons->variable_id = timer_id;

    return cons;
}

constraint* get_constraint(const string& exprs, expr* v_one, expr* v_two)
{
    constraint* cons = new constraint();
    if(exprs.find("<=") != std::string::npos)
        cons->operand = constraint::less_equal_c;
    else if(exprs.find(">=") != std::string::npos)
        cons->operand = constraint::greater_equal_c;
    else if(exprs.find("==") != std::string::npos)
        cons->operand = constraint::equal_c;
    else if(exprs.find("!=") != std::string::npos)
        cons->operand = constraint::not_equal_c;
    else if(exprs.find('<') != std::string::npos)
        cons->operand = constraint::less_c;
    else if(exprs.find('>') != std::string::npos)
        cons->operand = constraint::greater_c;
    else
    {
        THROW_LINE("Operand in " + exprs + " not found, sad..");
    }
    cons->uses_variable = false;
    cons->value = v_one;
    cons->expression = v_two;

    return cons;
}

expr* get_expression_con(const string& exprs, expr* left, expr* right)
{
    expr* expr_p = new expr();
    if(exprs.find("<=") != std::string::npos)
        expr_p->operand = expr::less_equal_ee;
    else if(exprs.find(">=") != std::string::npos)
        expr_p->operand = expr::greater_equal_ee;
    else if(exprs.find("==") != std::string::npos)
        expr_p->operand = expr::equal_ee;
    else if(exprs.find("!=") != std::string::npos)
        expr_p->operand = expr::not_equal_ee;
    else if(exprs.find('<') != std::string::npos)
        expr_p->operand = expr::less_ee;
    else if(exprs.find('>') != std::string::npos)
        expr_p->operand = expr::greater_ee;
    else
    {
        THROW_LINE("Operand in " + exprs + " not found, sad..");
    }

    expr_p->left = left;
    expr_p->right = right;

    return expr_p;
}

template<typename T>
arr<T> to_array(std::list<T>* list)
{
    int size = static_cast<int>(list->size());
    if(size == 0) return arr<T>::empty();
    T* array = static_cast<T*>(malloc(sizeof(T)*size));

    int i = 0;
    for(T item : *list)
    {
        array[i] = item;
        i++;
    }

    return arr<T>{array, size};
}


int uppaal_xml_parser::get_timer_id(const string& expr) const
{
    const string expr_wout_spaces = replace_all(remove_whitespace(expr), "\n", "");
    int index = 0;

    while (true)
    {
        if (static_cast<int>(expr.size()) == index)
        {
            break;
        }
        
        if (in_array(expr_wout_spaces[++index], {'<','>','='}))
        {
            break;
        }
    }

    const string sub = expr_wout_spaces.substr(0, index);
    if ( vars_map_.count(sub))
    {
        return vars_map_.at(sub);
    }
    
    if (global_vars_map_.count(sub))
    {
        return global_vars_map_.at(sub);
    }
    
    THROW_LINE("sum tin wong");
}

template <typename T>
void uppaal_xml_parser::fill_expressions(const list<string>& expressions, list<T>* t)
{
    for(const auto& expr: expressions)
    {
        if (expr.empty())
            continue;
        
        const extract_condition extracted_condition = string_extractor::extract(extract_condition(expr));

        if (extracted_condition.input.empty())
            continue;
        
        //TODO fix this plz
        //Constraint is heap allocated, and is then copied here.
        //Results in dead memory.
        
        if (timers_map_.count(extracted_condition.left) || global_timers_map_.count(extracted_condition.left))
            t->push_back(*get_constraint(extracted_condition.input,
                get_timer_id(extracted_condition.input),
                variable_expression_evaluator::evaluate_variable_expression(extracted_condition.right, &vars_map_, &global_vars_map_)));
        else
            t->push_back(*get_constraint(extracted_condition.input,
                variable_expression_evaluator::evaluate_variable_expression(extracted_condition.left, &vars_map_, &global_vars_map_),
                variable_expression_evaluator::evaluate_variable_expression(extracted_condition.right, &vars_map_, &global_vars_map_)));
    }
}


void uppaal_xml_parser::init_global_clocks(const xml_document* doc)
{
    string global_decl = doc->child("nta").child("declaration").child_value();
    global_decl = remove_whitespace(global_decl);
    const list<declaration> decls = dp_.parse(global_decl);
    for (declaration d : decls)
    {
        //global declarations
        if(d.get_type() == clock_type)
        {
            insert_to_map(&this->global_vars_map_, d.get_name(), vars_id_);
            insert_to_map(&this->global_timers_map_, d.get_name(), vars_id_);
            
            clock_var c_v = clock_var();
            c_v.id = vars_id_++;
            c_v.value = d.get_value();
            c_v.max_value = c_v.value;
            c_v.temp_value = c_v.value;
            c_v.rate = 1;
            c_v.should_track = false;
            vars_list_->push_back(c_v);
            
        }
        else if(d.get_type() == chan_type)
        {
            insert_to_map(&this->global_vars_map_, d.get_name(), chan_id_++);
        }
        else
        {
            insert_to_map(&this->global_vars_map_, d.get_name(), vars_id_);
            clock_var c_v = clock_var();
            c_v.id = vars_id_++;
            c_v.value = d.get_value();
            c_v.max_value = c_v.value;
            c_v.temp_value = c_v.value;
            c_v.rate = 1;
            c_v.should_track = false;
            vars_list_->push_back(c_v);

        }
    }
}

void uppaal_xml_parser::init_local_clocks(xml_node template_node)
{
    string decl = template_node.child("declaration").child_value();
    decl = remove_whitespace(decl);
    
    for (declaration d : dp_.parse(decl))
    {
        //local declarations
        if(d.get_type() == clock_type)
        {
            insert_to_map(&this->vars_map_, d.get_name(), vars_id_);
            insert_to_map(&this->timers_map_, d.get_name(), vars_id_);
            clock_var c_v = clock_var();
            c_v.id = vars_id_++;
            c_v.value = d.get_value();
            c_v.max_value = c_v.value;
            c_v.temp_value = c_v.value;
            c_v.rate = 1;
            c_v.should_track = false;
            vars_list_->push_back(c_v);
        }
        else if(d.get_type() == chan_type)
        {
            insert_to_map(&this->vars_map_, d.get_name(), chan_id_++);
        }
        else
        {
            insert_to_map(&this->vars_map_, d.get_name(), vars_id_);
            clock_var c_v = clock_var();
            c_v.id = vars_id_++;
            c_v.value = d.get_value();
            c_v.max_value = c_v.value;
            c_v.temp_value = c_v.value;
            c_v.rate = 0;
            c_v.should_track = true;
            vars_list_->push_back(c_v);

        }
    }
}

uppaal_xml_parser::uppaal_xml_parser()
= default;

node* uppaal_xml_parser::get_node(const int target_id, const list<node*>* arr) const
{
    for(node* node: *arr)
    {
        if(node->id == target_id)
            return node;
    }
    return arr->front();
}

void uppaal_xml_parser::handle_locations(const xml_node locs)
{
    const string string_id = locs.attribute("id").as_string();
    const string string_name = locs.child("name").child_value();
    const int node_id = string_extractor::extract(extract_node_id(string_id));
    list<constraint> invariants;
    expr* expo_rate = new expr;
    expo_rate->operand = expr::literal_ee;
    expo_rate->value = 1.0;
    bool is_goal = false;

    insert_to_map(&node_edge_map, node_id, list<edge>());
            
    if (!string_name.empty())
    {
        is_goal = string_name == "Goal" ? true : false;
        node_names_->emplace(node_id, string_name);
    }

    const string kind = locs.child("label").attribute("kind").as_string();
    const string expr_string = locs.child("label").child_value();
    const list<string> expressions = split_expr(expr_string);
            
    if (kind == "exponentialrate")
    {
        //TODO make string trimmer
        const string line_wo_ws = remove_whitespace(expr_string);
        const string nums = take_after(line_wo_ws, "=");

        delete expo_rate;
        expo_rate = variable_expression_evaluator::evaluate_variable_expression(nums,&vars_map_, &global_vars_map_);
    }
            
    if (kind == "invariant")
    {
        fill_expressions(expressions, &invariants);
    }
            
    if (init_node_id_ == node_id)
        start_nodes_.push_back(nodes_->size());
            
    node* node_ = new node();
    node_->id = node_id;
    node_->invariants = to_array(&invariants);
    node_->is_branch_point = false;
    node_->is_goal = is_goal;
    node_->lamda = expo_rate;
    node_->edges = arr<edge>::empty();
    
    nodes_->push_back(node_);
    nodes_map_->emplace(node_->id, node_with_system_id(node_, this->system_count_));
}

void uppaal_xml_parser::handle_transitions(const xml_node trans)
{
    const string source = trans.child("source").attribute("ref").as_string();
    const string target = trans.child("target").attribute("ref").as_string();

    const int source_id = string_extractor::extract(extract_node_id(source));
    const int target_id = string_extractor::extract(extract_node_id(target));
    
    list<constraint> guards;
    list<update> updates;
    expr* probability = nullptr;
    int ec = TAU_CHANNEL;
    
    for (pugi::xml_node labels: trans.children("label"))
    {
        string kind = labels.attribute("kind").as_string();
        string expr_string = labels.child_value();
        
        if(kind == "guard")
        {
            list<string> expressions = split_expr(expr_string);
            fill_expressions(expressions, &guards);
        }
        else if (kind == "assignment")
        {
            updates = handle_assignment(expr_string);
        }
        else if (kind == "synchronisation")
        {
            ec = handle_sync(expr_string);
        }
        else if (kind == "probability")
        {
            extract_probability extracted_probability = string_extractor::extract(extract_probability(expr_string));
            probability = variable_expression_evaluator::evaluate_variable_expression(extracted_probability.value,&vars_map_, &global_vars_map_);
        }
    }
    
    if (probability == nullptr)
    {
        probability = new expr();
        probability->operand = expr::literal_ee;
        probability->value = 1.0;
    }
    
    node* target_node = get_node(target_id, nodes_);
    edge result_edge = edge{
        ec,
        probability,
        target_node,
        to_array(&guards),
        to_array(&updates)
    };
    
    node_edge_map.at(source_id).push_back(result_edge);
}

bool uppaal_xml_parser::is_if_statement(const string& expr)
{
    return expr.find("?")!=std::string::npos && expr.find(":")!=std::string::npos;
}

expr* uppaal_xml_parser::handle_if_statement(const string& input)
{
    const extract_if_statement extracted_if_statement = string_extractor::extract(extract_if_statement(input));
    const extract_condition extracted_condition = string_extractor::extract(extract_condition(extracted_if_statement.condition));

    //Build the condition
    expr* right_side_con_expr = variable_expression_evaluator::evaluate_variable_expression(extracted_condition.right,&this->vars_map_, &this->global_vars_map_);
    expr* left_side_con_expr = variable_expression_evaluator::evaluate_variable_expression(extracted_condition.left,&this->vars_map_, &this->global_vars_map_);
    expr* condition_e = get_expression_con(extracted_if_statement.condition, left_side_con_expr, right_side_con_expr);
    
    expr* if_true_e = variable_expression_evaluator::evaluate_variable_expression(extracted_if_statement.if_true,&this->vars_map_, &this->global_vars_map_);
    expr* if_false_e = variable_expression_evaluator::evaluate_variable_expression(extracted_if_statement.if_false,&this->vars_map_, &this->global_vars_map_);

    expr* whole_expr = new expr();
    whole_expr->left = condition_e;
    whole_expr->right = if_true_e;
    whole_expr->conditional_else = if_false_e;
    
    return whole_expr;
}

list<update> uppaal_xml_parser::handle_assignment(const string& input)
{
    const list<string> expressions = split_expr(input, ',');
    list<update> result;
    expr* right_side;
    
    for(const auto& expr: expressions)
    {

        extract_assignment extracted_assignment = string_extractor::extract(extract_assignment(expr));

        if (extracted_assignment.input.empty())
            continue;

        if (is_if_statement(extracted_assignment.right))
        {
            //Is if statement
            right_side = handle_if_statement(extracted_assignment.right);
        }
        else
        {
            //Is normal assignment
            right_side = variable_expression_evaluator::evaluate_variable_expression(extracted_assignment.right,&this->vars_map_, &this->global_vars_map_);
        }

        update upd;
        upd.variable_id = get_timer_id(extracted_assignment.left);
        upd.expression = right_side;
        
        result.emplace_back(upd);
    }
    
    return result;
}

int uppaal_xml_parser::handle_sync(const string& input) const
{
    int ec = TAU_CHANNEL;

    const extract_sync extracted_sync = string_extractor::extract(extract_sync(input));

    if (vars_map_.count(extracted_sync.keyword))
        ec = vars_map_.at(extracted_sync.keyword);
    else if (global_vars_map_.count(extracted_sync.keyword))
        ec = global_vars_map_.at(extracted_sync.keyword);
    else
    {
        THROW_LINE(extracted_sync.keyword + " NOT IN LOCAL, NOR GLOBAL MAP, comeon dude..");
    }
    
    return extracted_sync.is_listener ? -ec : ec;
}

arr<node*> uppaal_xml_parser::after_processing()
{
    for(node* node: *nodes_)
    {
        node->edges = to_array(&node_edge_map.at(node->id));
    }
    auto size = start_nodes_.size();
    const arr<node*> start_nodes = arr<node*>{static_cast<node**>(malloc(sizeof(node*) * size)), static_cast<int>(size)};

    int number_of_start_nodes = 0;
    for (const int i : start_nodes_)
    {
        list<node*>::iterator n_front = nodes_->begin();
        std::advance(n_front, i);
        start_nodes.store[number_of_start_nodes++] = *n_front;
    }
    printf("SIZE %d\n", start_nodes.size);
    
    return start_nodes;
}


__host__ automata uppaal_xml_parser::parse_xml(const char* file_path)
{
    string path = file_path;
    xml_document doc;
    declaration_parser dp;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        THROW_LINE("The specified file does not exist.. stupid.")
    }
    
    
    init_global_clocks(&doc);
    
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        const string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = string_extractor::extract(extract_node_id(init_node));
        init_local_clocks(templates);
        
        
        for (const pugi::xml_node locs: templates.children("location"))
        {
            handle_locations(locs);
        }
        
        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            const string string_id = locs.attribute("id").as_string();
            const int node_id = string_extractor::extract(extract_node_id(string_id));
            insert_to_map(&node_edge_map, node_id, list<edge>());
            expr* lamda = new expr;
            lamda->operand = expr::literal_ee;
            lamda->value = 1.0;

            node* node_ = new node();
            node_->id = node_id;
            node_->invariants = arr<constraint>::empty();
            node_->is_branch_point = true;
            node_->is_goal = false;
            node_->lamda = lamda;
            node_->edges = arr<edge>::empty();
            
            nodes_->push_back(node_);
        }

        
        
        for (const pugi::xml_node trans: templates.children("transition"))
        {
            handle_transitions(trans);
        }
        
        //Done with system, clear storage
        vars_map_.clear();
        timers_map_.clear();
        this->system_count_++;
    }
    
    const auto start_nodes = after_processing();
    return automata{start_nodes, to_array(vars_list_)};
}

__host__ automata uppaal_xml_parser::parse(string file_path)
{
    try
    {
        char* writeable = new char[file_path.size() + 1];
        std::copy(file_path.begin(), file_path.end(), writeable);
        writeable[file_path.size()] = '\0';
        auto model = parse_xml(writeable);
        delete[] writeable;
        return model;
    }
    catch (const std::runtime_error &ex)
    {
        throw runtime_error("parse error");
    }
}

bool uppaal_xml_parser::try_parse_block_threads(const std::string& str, unsigned* out_blocks, unsigned* out_threads)
{
    list<string> split = helper::split_all(str, ",");
    if(split.size() != 2) return false;
    const string blocks = split.front();
    const string threads = split.back();

    try
    {
        *out_blocks  = static_cast<unsigned>(stoi(blocks));
        *out_threads = static_cast<unsigned>(stoi(threads));
    }
    catch(invalid_argument&)
    {
        return false;
    }
    return true;
}

bool uppaal_xml_parser::try_parse_units(const std::string& str, bool* is_time, double* value)
{
    const char unit = str.back();
    if(unit != 's' && unit != 't') return false;
    *is_time = unit == 't';

    const string val = str.substr(0, str.size() - 1);

    try
    {
        *value = stod(val);
    }
    catch(invalid_argument&)
    {
        return false;
    }
    return true;
}
