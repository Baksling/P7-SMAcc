#include "uppaal_tree_parser.h"



constraint_t* get_constraint(const string& expr, const int timer_id, const float value)
{
    if(expr.find("<=") != std::string::npos)
        return constraint_t::less_equal_v(timer_id, value);
    if(expr.find(">=") != std::string::npos)
        return constraint_t::greater_equal_v(timer_id, value);
    if(expr.find("==") != std::string::npos)
        return constraint_t::equal_v(timer_id, value);
    if(expr.find('<') != std::string::npos)
        return constraint_t::less_v(timer_id, value);
    if(expr.find('>') != std::string::npos)
        return constraint_t::greater_v(timer_id, value);
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

void uppaal_tree_parser::init_clocks(const xml_document* doc)
{
    declaration_parser dp;
    string global_decl = doc->child("nta").child("declaration").child_value();
    cout << "\nGLOBAL GUYS: " << global_decl << " :NICE\n";
    global_decl = replace_all(global_decl, " ", "");
    const list<declaration> decls = dp.parse(global_decl);
    cout << "\nSIZE: " << decls.size() << "\n";
        
    for (declaration d : decls)
    {
        //global declarations
        if(d.get_type() == clock_type)
        {
            global_vars_map_.insert_or_assign(d.get_name(),clock_id_);
            timer_list_.push_back(new clock_variable(clock_id_++, d.get_value()));
        }
        else
        {
            global_vars_map_.insert_or_assign(d.get_name(), var_id_);
            var_list_.push_back(new clock_variable(var_id_++, d.get_value()));
        }
    }
    
    
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        string decl = templates.child("declaration").child_value();
        decl = replace_all(decl, " ", "");
        list<declaration> declarations = dp.parse(decl);
        cout << "\nSIZE: " << declarations.size() << "\n";
        
        for (declaration d : declarations)
        {
            //local declarations
            if(d.get_type() == clock_type)
            {
                vars_map_.insert_or_assign(d.get_name(),clock_id_);
                timer_list_.push_back(new clock_variable(clock_id_++, d.get_value()));
            }
            else
            {
                vars_map_.insert_or_assign(d.get_name(), var_id_);
                var_list_.push_back(new clock_variable(var_id_++, d.get_value()));
            }
        }
    }
}

uppaal_tree_parser::uppaal_tree_parser()
= default;

node_t* uppaal_tree_parser::get_node(const int target_id) const
{
    for(node_t* node: *nodes_)
    {
        if(node->get_id() == target_id)
            return node;
    }
    return nodes_->front();
}


__host__ stochastic_model_t uppaal_tree_parser::parse_xml(char* file_path)
{
    string path = file_path;
    xml_document doc;
    map<int, list<edge_t*>> node_edge_map;
    
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
        for (pugi::xml_node locs: templates.children("location"))
        {
            string string_id = locs.attribute("id").as_string();
            string string_name = locs.child("name").child_value();
            const int node_id = xml_id_to_int(string_id);
            bool is_goal = false;
            node_edge_map.insert_or_assign(node_id, list<edge_t*>());
            
            list<constraint_t*> invariants;
            expression* expo_rate = nullptr;
            
            if (string_name == "Goal")
                is_goal = true;
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr_string = locs.child("label").child_value();

            list<string> expressions = split_expr(expr_string);
            
            if (kind == "exponentialrate")
            {
                expo_rate = update_parser::parse(expr_string, &vars_map_, &global_vars_map_);
            }
            
            if (kind == "invariant")
            {
                for(const auto& expr: expressions)
                {
                    if (expr.empty())
                        continue;
                    invariants.push_back(get_constraint(expr, get_timer_id(expr), get_expr_value_float(expr)));
                }
            }
            nodes_->push_back(new node_t(node_id, to_array(&invariants),false, is_goal, expo_rate));
        }

        for (pugi::xml_node locs: templates.children("branchpoint"))
        {
            string string_id = locs.attribute("id").as_string();
            const int node_id = xml_id_to_int(string_id);
            node_edge_map.insert_or_assign(node_id, list<edge_t*>());
            nodes_->push_back(new node_t(node_id,array_t<constraint_t*>(0), true));
        }

        string init_node = templates.child("init").attribute("ref").as_string();
        init_node_id_ = xml_id_to_int(init_node);
        
        for (pugi::xml_node trans: templates.children("transition"))
        {
            string source = trans.child("source").attribute("ref").as_string();
            string target = trans.child("target").attribute("ref").as_string();

            int source_id = xml_id_to_int(source);
            int target_id = xml_id_to_int(target);
            
            list<constraint_t*> guards;
            list<update_t*> updates;
            float probability = 1.0f;
            
            for (pugi::xml_node labels: trans.children("label"))
            {
                string kind = labels.attribute("kind").as_string();
                string expr_string = labels.child_value();

                
                if(kind == "guard")
                {
                    list<string> expressions = split_expr(expr_string);
                    for(const auto& expr: expressions)
                    {
                        if (expr.empty())
                            continue;
                        guards.push_back(get_constraint(expr,get_timer_id(expr), get_expr_value_float(expr)));
                    }
                }
                else if (kind == "assignment")
                {
                    list<string> expressions = split_expr(expr_string, ',');
                    cout << "\nASS0: " << expressions.size() << " " << expr_string <<"\n";
                    for(const auto& expr: expressions)
                    {
                        if (expr.empty())
                            continue;
                        cout << "\nASS: " << expr <<"\n";
                        updates.push_back(new update_t(update_id++, get_timer_id(expr), true, update_parser::parse(expr, &vars_map_, &global_vars_map_)));
                        cout << "\nIT OK: " << expr <<"\n";
                        cout.flush();
                    }
                }
                else if (kind == "probability")
                {
                    probability = get_expr_value_float(expr_string);
                }
            }
            cout << "\nIT OK:2 "  <<"\n";
            cout.flush();
            node_t* target_node = get_node(target_id);
            auto result_edge = new edge_t(edge_id++, probability, target_node, to_array(&guards), to_array(&updates));
            cout << "guard size: " << guards.size() << "\n";
            
            // if (guards.empty())
            //     result_edge = new edge_t(edge_id, probability, target_node, array_t<constraint_t*>(0));
            //result_edge->set_updates(&updates);
            
            node_edge_map.at(source_id).push_back(result_edge);
        }
        vars_map_.clear();
    }

    for(node_t* node: *nodes_)
    {
        node->set_edges(&node_edge_map.at(node->get_id()));
    }

    return stochastic_model_t(get_node(init_node_id_), to_array(&timer_list_), to_array(&var_list_));
}

__host__ stochastic_model_t uppaal_tree_parser::parse(char* file_path)
{
    try
    {
        return parse_xml(file_path);
    }
    catch (const std::runtime_error &ex)
    {
        cout << "Parse error: " << ex.what() << "\n";
        throw runtime_error("parse error");
    }
}