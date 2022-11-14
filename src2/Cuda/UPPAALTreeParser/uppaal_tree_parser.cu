#include "uppaal_tree_parser.h"


#define ALPHA "abcdefghijklmnopqrstuvwxyz"



char get_constraint_op(const string& expr)
{
    if(expr.find("<=") != std::string::npos)
        return '=';
    if(expr.find(">=") != std::string::npos)
        return '=';
    if(expr.find("==") != std::string::npos)
        return '=';
    if(expr.find("!=") != std::string::npos)
        return '=';
    if(expr.find('<') != std::string::npos)
        return '<';
    if(expr.find('>') != std::string::npos)
        return '>';
    THROW_LINE("Operand in " + expr + " not found, sad..");
}

constraint_t* get_constraint(const string& expr, const int timer_id, expression* value)
{
    if(expr.find("<=") != std::string::npos)
        return constraint_t::less_equal_v(timer_id, value);
    if(expr.find(">=") != std::string::npos)
        return constraint_t::greater_equal_v(timer_id,value);
    if(expr.find("==") != std::string::npos)
        return constraint_t::equal_v(timer_id,value);
    if(expr.find("!=") != std::string::npos)
        return constraint_t::not_equal_v(timer_id,value);
    if(expr.find('<') != std::string::npos)
        return constraint_t::less_v(timer_id,value);
    if(expr.find('>') != std::string::npos)
        return constraint_t::greater_v(timer_id,value);
    THROW_LINE("Operand in " + expr + " not found, sad..");
}

constraint_t* get_constraint(const string& expr, const int timer_id_1, const int timer_id_2)
{
    if(expr.find("<=") != std::string::npos)
        return constraint_t::less_equal_t(timer_id_1, timer_id_2);
    if(expr.find(">=") != std::string::npos)
        return constraint_t::greater_equal_t(timer_id_1, timer_id_2);
    if(expr.find("==") != std::string::npos)
        return constraint_t::equal_t(timer_id_1, timer_id_2);
    if(expr.find('<') != std::string::npos)
        return constraint_t::less_t(timer_id_1, timer_id_2);
    if(expr.find('>') != std::string::npos)
        return constraint_t::greater_t(timer_id_1, timer_id_2);
    THROW_LINE("Operand in " + expr + " not found, sad..")
}


template <typename T> T* list_to_arr(list<T> l)
{
    T* arr = static_cast<T*>(malloc(sizeof(T) * l.size()));
    int k = 0;
    for (T const &i: l) {
        arr[k++] = i;
    }
    
    return arr;
}


int uppaal_tree_parser::get_timer_id(const string& expr) const
{
    const string expr_wout_spaces = replace_all(expr, string(" "), string(""));
    int index = 0;

    while (true)
    {
        if (static_cast<int>(expr.size()) == index)
        {
            THROW_LINE("sum tin wong")
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
    
    THROW_LINE("sum tin wong")
}

template <typename T>
void uppaal_tree_parser::get_guys(const list<string>& expressions, list<T>* t)
{
    for(const auto& expr: expressions)
    {
        if (expr.empty())
            continue;

        const string right_side = take_after(expr, get_constraint_op(expr));
        //TODO fix this plz
        //Constraint is heap allocated, and is then copied here.
        //Results in dead memory.
        t->push_back(*get_constraint(expr, get_timer_id(expr), update_parser::parse(right_side, &vars_map_, &global_vars_map_)));
    }
}


void uppaal_tree_parser::init_clocks(const xml_document* doc)
{
    string global_decl = doc->child("nta").child("declaration").child_value();
    global_decl = replace_all(global_decl, " ", "");
    const list<declaration> decls = dp_.parse(global_decl);
        
    for (declaration d : decls)
    {
        //global declarations
        if(d.get_type() == clock_type)
        {
            global_vars_map_.insert_or_assign(d.get_name(),clock_id_);
            timers_map_.insert_or_assign(d.get_name(), clock_id_);
            timer_list_->push_back(clock_variable(clock_id_++, d.get_value()));
        }
        else if(d.get_type() == chan_type)
        {
            global_vars_map_.insert_or_assign(d.get_name(), chan_id_++);
        }
        else
        {
            global_vars_map_.insert_or_assign(d.get_name(), var_id_);
            var_list_->push_back(clock_variable(var_id_++, d.get_value()));
        }
    }
    
    
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        string decl = templates.child("declaration").child_value();
        decl = replace_all(decl, " ", "");
        list<declaration> declarations = dp_.parse(decl);
        
        for (declaration d : declarations)
        {
            //local declarations
            if(d.get_type() == clock_type)
            {
                vars_map_.insert_or_assign(d.get_name(),clock_id_);
                timers_map_.insert_or_assign(d.get_name(), clock_id_);
                timer_list_->push_back(clock_variable(clock_id_++, d.get_value()));
            }
            else if(d.get_type() == chan_type)
            {
                vars_map_.insert_or_assign(d.get_name(), chan_id_++);
            }
            else
            {
                vars_map_.insert_or_assign(d.get_name(), var_id_);
                var_list_->push_back(clock_variable(var_id_++, d.get_value()));
            }
        }
    }
}

uppaal_tree_parser::uppaal_tree_parser()
= default;

node_t* uppaal_tree_parser::get_node(const int target_id, const list<node_t*>* arr) const
{
    for(node_t* node: *arr)
    {
        if(node->get_id() == target_id)
            return node;
    }
    return arr->front();
}


__host__ stochastic_model_t uppaal_tree_parser::parse_xml(char* file_path)
{
    string path = file_path;
    xml_document doc;
    unordered_map<int, list<edge_t>> node_edge_map; //TODO edge is no longer pointer
    declaration_parser dp;
    
    // load the XML file
    if (!doc.load_file(file_path))
    {
        THROW_LINE("The specified file does not exist.. stupid.")
    }

    int edge_id = 0;
    int update_id = 0;

    init_clocks(&doc);
    
    for (pugi::xml_node templates: doc.child("nta").children("template"))
    {
        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            string string_name = locs.child("name").child_value();
            const int node_id = xml_id_to_int(string_id);
            bool is_goal = false;
            node_edge_map.insert_or_assign(node_id, list<edge_t>());
            
            list<constraint_t> invariants; //TODO no longer pointer
            expression* expo_rate = nullptr;
            
            if (string_name != "")
            {
                is_goal = string_name == "Goal" ? true : false;
                node_names_->emplace(node_id, string_name);
            }
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr_string = locs.child("label").child_value();

            list<string> expressions = split_expr(expr_string);
            
            if (kind == "exponentialrate")
            {
                expo_rate = update_parser::parse(expr_string, &vars_map_, &global_vars_map_);
            }
            
            if (kind == "invariant")
            {
                get_guys(expressions, &invariants);
            }
            if (init_node_id_ == node_id)
                start_nodes_.push_back(nodes_->size());
            
            node_t* node = new node_t(node_id, to_array(&invariants),false, is_goal, expo_rate);
            nodes_->push_back(node);
            nodes_map_->emplace(node->get_id(), node_with_system_id(node, this->system_count));
        }

        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            string string_id = locs.attribute("id").as_string();
            const int node_id = xml_id_to_int(string_id);
            node_edge_map.insert_or_assign(node_id, list<edge_t>()); //TODO removed pointer edge pointer
            nodes_->push_back(new node_t(node_id,array_t<constraint_t>(0), true));
        }

        
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            string source = trans.child("source").attribute("ref").as_string();
            string target = trans.child("target").attribute("ref").as_string();

            int source_id = xml_id_to_int(source);
            int target_id = xml_id_to_int(target);
            
            list<constraint_t> guards; //TODO removed pointer part
            list<update_t> updates; //TODO removed pointer part
            expression* probability = nullptr;
            edge_channel* ec = nullptr;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr_string = labels.child_value();

                
                if(kind == "guard")
                {
                    list<string> expressions = split_expr(expr_string);
                    get_guys(expressions, &guards);
                }
                else if (kind == "assignment")
                {
                    list<string> expressions = split_expr(expr_string, ',');
                    for(const auto& expr: expressions)
                    {
                        if (expr.empty())
                            continue;
                        
                        expression* e = update_parser::parse(expr, &vars_map_, &global_vars_map_);
                        string keyword = take_while(expr, '=');
                        bool is_clock = false;

                        if(timers_map_.count(keyword) > 0)
                        {
                            is_clock = true;
                        }

                        
                        updates.push_back(update_t(update_id++, get_timer_id(expr), is_clock, e));
                    }
                }
                else if (kind == "synchronisation")
                {
                    ec = new edge_channel();
                    // printf("\n !=!=!==!=!=!== %s \n | %d", expr_string.c_str(), !does_not_contain(expr_string, "!"));
                    if (!does_not_contain(expr_string, "!"))
                    {
                        ec->is_listener = false;
                    }
                    else
                    {
                        ec->is_listener = true;
                    }
                    string sync_keyword = replace_all(expr_string, "!", "");
                    sync_keyword = replace_all(sync_keyword, "?", "");
                    sync_keyword = replace_all(sync_keyword, " ", "");
                    
                    if (vars_map_.count(sync_keyword))
                    {
                        ec->channel_id = vars_map_.at(sync_keyword);
                    }
                    else if (global_vars_map_.count(sync_keyword))
                    {
                        ec->channel_id = global_vars_map_.at(sync_keyword);
                    }
                    else
                    {
                        THROW_LINE(sync_keyword + " NOT IN LOCAL, NOR GLOBAL MAP, comeon dude..");
                    }
                    
                }
                else if (kind == "probability")
                {
                    probability = update_parser::parse(expr_string, &vars_map_, &global_vars_map_);
                }
            }

            if (probability == nullptr) probability = expression::literal_expression(1.0);
            
            node_t* target_node = get_node(target_id, nodes_);
            edge_t result_edge = ec == nullptr
                ? edge_t(edge_id++, probability, target_node, to_array(&guards), to_array(&updates))
                : edge_t(edge_id++, probability, target_node, to_array(&guards), to_array(&updates), *ec);
            
            node_edge_map.at(source_id).push_back(result_edge);
        }
        vars_map_.clear();
        this->system_count++;
    }

    for(node_t* node: *nodes_)
    {
        node->set_edges(&node_edge_map.at(node->get_id()));
    }
    
    //TODO i broke plz fix :)
    //The stochastic model now expects an array of objects, rather than a array of object pointers.
    //This helps cut down on the number of times pointers need to be followed in the simulation.
    //Only reason it was like that before, was because we didnt know how to make the cuda-allocation code without it :)
    // - Bak ඞ
    array_t<node_t*> start_nodes = array_t<node_t*>(start_nodes_.size());

    

    int number_of_start_nodes = 0;
    for (int i : start_nodes_)
    {
        auto n_front = nodes_->begin();
        std::advance(n_front, i);
        start_nodes.arr()[number_of_start_nodes++] = *n_front;
    }

    return stochastic_model_t(start_nodes, to_array(timer_list_), to_array(var_list_));
}

__host__ stochastic_model_t uppaal_tree_parser::parse(string file_path)
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
        cout << "Parse error: " << ex.what() << "\n";
        throw runtime_error("parse error");
    }
}