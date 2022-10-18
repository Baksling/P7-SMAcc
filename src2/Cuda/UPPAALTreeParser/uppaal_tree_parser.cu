#include "uppaal_tree_parser.h"
//#include "../Domain/uneven_list.h"


#define GPU __device__
#define CPU __host__

using namespace std;
using namespace pugi;

#define THROW_LINE(arg); throw parser_exception(arg, __FILE__, __LINE__);

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

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

bool in_array(const char &value, const std::vector<char> &array)
{
    return std::find(array.begin(), array.end(), value) != array.end();
}

int get_expr_value(const string& expr)
{
    const string expr_wo_ws = replace_all(expr, " ", "");
    unsigned long long index = expr_wo_ws.length();
    while (true)
    {
        if (index == 0)
        {
            return 0;
        }
        
        if (!in_array(expr_wo_ws[--index], {'1','2','3','4','5','6','7','8','9','0'}))
        {
            return stoi(expr_wo_ws.substr(index+1));
        }
    }
}

float get_expr_value_float(const string& expr)
{
    const string expr_wo_ws = replace_all(expr, " ", "");
    unsigned long long index = expr_wo_ws.length();
    while (true)
    {
        if (index == 0)
        {
            return stof(expr_wo_ws);
        }
        
        if (in_array(expr_wo_ws[--index], {'=',' ','<','>'}))
        {
            return stof(expr_wo_ws.substr(index+1));
        }
    }
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

int xml_id_to_int(string id_string)
{
    return stoi(id_string.replace(0,2,""));
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

    if ( timers_map_.count(sub) == 0)
    {
        THROW_LINE("sum tin wong")
    }
    
    return timers_map_.at(sub);
}

list<string> split_expr(const string& expr)
{
    list<string> result;
    std::stringstream test(expr);
    std::string segment;
    while(std::getline(test, segment, '&'))
    {
        string s = replace_all(segment, "&", "");
        result.push_back(s);
    }
    return result;
}

void uppaal_tree_parser::init_clocks(const xml_document* doc)
{
    for (pugi::xml_node templates: doc->child("nta").children("template"))
    {
        string decl = templates.child("declaration").child_value();

        const size_t clock_start = decl.find("clock");

        const size_t clock_end = decl.find(';');
        
        string clocks = decl.substr(clock_start+5,clock_end-1);
        clocks = replace_all(clocks, string(" "), string(""));

        std::stringstream clocks_stream(clocks);
        std::string clock;
        
        int var_amount = 0;
        while(std::getline(clocks_stream, clock, ','))
        {
            string clock_without = clock;
            if (clock.find(';') != std::string::npos)
                clock_without = replace_all(clock, ";", "");
            timers_map_.insert_or_assign(clock_without,var_amount++);
        }
        
        timer_amount_ = var_amount;
        cout << "INIT VAR AMOUNT SIZE: " << var_amount << "\n";
        for (int i = 0; i < var_amount; i++)
        {
            cout << "INIT CLOCK WITH ID: " << i << "\n";
            timer_list_.push_back(new clock_variable(i, 0));
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

template <typename T> void insert_into_list(list<list<T>>* t_list, int index, T item)
{
    auto l_front = t_list->begin();
    std::advance(l_front, index);
    l_front->emplace_back(item);
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
            
            if (string_name == "Goal")
                is_goal = true;
            
            string kind = locs.child("label").attribute("kind").as_string();
            string expr_string = locs.child("label").child_value();

            list<string> expressions = split_expr(expr_string);
            if (kind == "invariant")
            {
                for(const auto& expr: expressions)
                {
                    if (expr.empty())
                        continue;
                    invariants.push_back(get_constraint(expr, get_timer_id(expr), get_expr_value_float(expr)));
                }
            }
            if (invariants.empty())
                nodes_->push_back(new node_t(node_id, array_t<constraint_t*>(0), false, is_goal));
            else
                nodes_->push_back(new node_t(node_id, to_array(&invariants),false, is_goal));
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

                list<string> expressions = split_expr(expr_string);
                if(kind == "guard")
                {
                    for(const auto& expr: expressions)
                    {
                        if (expr.empty())
                            continue;
                        guards.push_back(get_constraint(expr,get_timer_id(expr), get_expr_value_float(expr)));
                    }
                }
                else if (kind == "assignment")
                {
                    for(const auto& expr: expressions)
                    {
                        if (expr.empty())
                            continue;
                        // updates.push_back(new update_t(update_id++, get_timer_id(expr), true, get_expr_value_float(expr)));
                    }
                }
                else if (kind == "probability")
                {
                    probability = get_expr_value_float(expr_string);
                }
            }
            
            node_t* target_node = get_node(target_id);
            // auto result_edge = new edge_t(edge_id++, probability, target_node, to_array(&guards));
            // cout << "guard size: " << guards.size() << "\n";
            //
            // if (guards.empty())
            //     result_edge = new edge_t(edge_id, probability, target_node, array_t<constraint_t*>(0));
            //
            // result_edge->set_updates(&updates);
            // node_edge_map.at(source_id).push_back(result_edge);
        }
    }

    for(node_t* node: *nodes_)
    {
        node->set_edges(&node_edge_map.at(node->get_id()));
    }

    return stochastic_model_t(get_node(init_node_id_), to_array(&timer_list_), array_t<clock_variable*>(0));
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
