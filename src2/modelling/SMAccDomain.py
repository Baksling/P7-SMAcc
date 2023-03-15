from typing import Set, Tuple, Dict, Iterable, Iterator, List
import ast
from itertools import chain


class Visitor:
    pass


class ModelException(Exception):
    def __init__(self, err: str):
        super().__init__(err)


class Visitable:
    def visit(self, visitor: Visitor):
        pass


TAU_CHANNEL = "TAU"


class Channel(Visitable):

    def __init__(self, channel_id: str = TAU_CHANNEL):
        self._channel_id = channel_id

    def _get_channel_id(self):
        return self._channel_id

    def as_directed(self, as_listener: bool):
        return DirectedChannel(self._channel_id, is_listener=as_listener)

    def as_listener(self):
        return DirectedChannel(self._channel_id, is_listener=True)

    def as_broadcast(self):
        return DirectedChannel(self._channel_id, is_listener=False)

    def __eq__(self, other):
        if isinstance(self, DirectedChannel) and isinstance(other, DirectedChannel):
            if self._channel_id == other._channel_id == TAU_CHANNEL:
                return True
            else: 
                return self._channel_id == other._get_channel_id() and self._listener == other.is_listener()
        if isinstance(other, Channel):
            return self._channel_id == other._channel_id
        return False

    def __hash__(self):
        return hash(self._channel_id)

    def visit(self, visitor: Visitor):
        return


class DirectedChannel(Channel):
    def __init__(self, channel_id: str, is_listener=True):
        super(Channel, self).__init__()
        self._channel_id = channel_id
        self._listener = is_listener

    def is_listener(self):
        return self._listener

    def is_broadcast(self):
        return not self._listener


def _default_convert_channel(ch: Channel | None) -> DirectedChannel | None:
    if ch is None:
        return DirectedChannel(TAU_CHANNEL, is_listener=False) 
    if isinstance(ch, DirectedChannel):
        return ch
    if isinstance(ch, Channel):
        return ch.as_broadcast()
    raise ModelException(f"object '{ch}' is not a channel")


class Expr(Visitable):
    def __init__(self, expr: str):
        super().__init__()
        self._expr = expr

    def get_expr_str(self):
        return self._expr

    def get_variables_ref(self) -> List[str]:
        return [node.id for node in ast.walk(ast.parse(self._expr))
                if isinstance(node, ast.Name)]

    def __str__(self):
        return self._expr

    def visit(self, visitor: Visitor):
        return

class ConditionalExpr(Expr):
    def __init__(self, condition: Expr | str, left: Expr | str, right: Expr | str):
        super().__init__(str(condition))
        self._cond = _to_expr_(condition)
        self._left = _to_expr_(left)
        self._right = _to_expr_(right)
    
    def get_expr_str(self):
        return f"{self._cond} ? {self._left} : {self._right}"
        
    def visit(self, visitor: Visitor):
        visitor.visit_expr(self._cond)
        visitor.visit_expr(self._left)
        visitor.visit_expr(self._right)
        

def _to_expr_(e: Expr | str) -> Expr:
    if isinstance(e, str):
        return Expr(e)
    if isinstance(e, Expr):
        return e
    raise ModelException(f"{e} could not be converted into expression, as it is neither an expression or string")


class Variable(Visitable):
    def __init__(self, name: str, init_value: float, rate: int = 1, track_value = True):
        super().__init__()
        if not name.isidentifier():
            raise ModelException(f"Variable {name} is not a valid variable name")
        self._name = name
        self._init_value = init_value
        self._rate = rate
        self._track_value = bool(track_value)

    def get_name(self):
        return self._name

    def get_init_value(self):
        return self._init_value
    
    def should_track(self):
        return self._track_value
    
    def get_rate(self):
        return self._rate
    
    def is_clock(self):
        return self._rate > 0

    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self._name == other._name
        elif isinstance(other, str):
            return self._name == other
        return False


def _validate_variables_(vars: Set[Variable]):
    seen = set()
    for v in vars:
        name = v.get_name()
        if name in seen:
            raise ModelException(f"variable {name} already exists in scope")
        seen.add(name)


def any_clocks_in_expr(expr: Expr, vars: Iterable[Variable]):
    var_map: Dict[str, Variable] = {v.get_name(): v for v in vars}
    for varname in [v for v in expr.get_variables_ref()]:
        var = var_map.get(varname, None)
        if var is not None and var.is_clock():
            print("vararara", var, var.is_clock(), var_map)
            return True
    return False


class Constraint(Visitable):
    allowed_types = {'>', '>=', '=', '!=', '<=', '<'}

    def __init__(self, left: Variable | Expr | str, type: str, right: Variable | Expr | str):
        super(Constraint, self).__init__()
        if type not in Constraint.allowed_types:
            raise ModelException(f"Constraint with invalid type '{type}' encountered. Must be one of the follwoing "
                                 f"tpyes {Constraint.allowed_types}")
        self._type = type
        self._left: Variable | Expr = left if isinstance(left, Variable) else _to_expr_(left)
        self._right: Variable | Expr = right if isinstance(right, Variable) else _to_expr_(right)

    def get_type(self):
        return self._type

    def get_left(self) -> Tuple[Variable | None, Expr | None]:
        if isinstance(self._left, Variable):
            return self._left, None
        else:
            return None, self._left

    def get_right(self) -> Tuple[Variable | None, Expr | None]:
        if isinstance(self._right, Variable):
            return self._right, None
        else:
            return None, self._right

    def validate(self, scope1: Set[Variable], scope2: Set[Variable]) -> int:
        if isinstance(self._left, Variable) and isinstance(self._right, Variable) and \
                self._left.is_clock() and self._right.is_clock():
            return False  # both sides contain clock
        if isinstance(self._left, Expr) and any_clocks_in_expr(self._left, chain(scope1, scope2)):
            return False  # left side contains clock
        if isinstance(self._right, Expr) and any_clocks_in_expr(self._right, chain(scope1, scope2)):
            return False  # right side contains clock
        return True

    def visit(self, visitor: Visitor):
        if isinstance(self._left, Variable):
            visitor.visit_variable(self._left)
        elif isinstance(self._left, Expr):
            visitor.visit_expr(self._left)
        if isinstance(self._right, Variable):
            visitor.visit_variable(self._right)
        elif isinstance(self._right, Expr):
            visitor.visit_expr(self._right)
        
    def __str__(self):
        return f"{self._left} {self._type} {self._right}"


class Update(Visitable):
    def __init__(self, var: Variable, expr: Expr | str):
        self._var = var
        self._expr = _to_expr_(expr)

    def get_variable(self):
        return self._var

    def get_expr(self):
        return self._expr

    def visit(self, visitor: Visitor):
        visitor.visit_variable(self._var)
        visitor.visit_expr(self._expr)


class Node(Visitable):
    default_roe = Expr("1")

    def __init__(self, name, *, exponential_rate: Expr | str = None, invariants: Set[Constraint] = set(), is_branch_node = False):
        exponential_rate = Node.default_roe if exponential_rate is None else exponential_rate
        self._name = name
        self._invariants = invariants
        self._rate_of_exponential = _to_expr_(exponential_rate)
        self._is_branch_node = is_branch_node

    def get_name(self):
        return self._name

    def get_invariants(self):
        return self._invariants

    def get_rate_of_exponential(self):
        return self._rate_of_exponential
    
    def is_branch_node(self):
        return self._is_branch_node

    def __str__(self):
        return self._name

    def visit(self, visitor: Visitor):
        visitor.visit_expr(self._rate_of_exponential)
        for c in self._invariants:
            visitor.visit_constraint(c)


class Edge(Visitable):
    default_weight = Expr("1")

    def __init__(self,
                 source: Node,
                 dest: Node,
                 actions: Set[Update | Constraint] = {},
                 weight: Expr | str = None,
                 channel: Channel = None):
        self._source = source
        self._dest = dest
        self._weight = _to_expr_(weight) if weight is not None else Edge.default_weight
        self._actions = actions
        self._channel = _default_convert_channel(channel)
        if any((1 for x in actions if not isinstance(x, Update) and not isinstance(x, Constraint))):
            raise ModelException("An edge with an action which is neither update nor constraint encountered.")

    def get_source_node(self):
        return self._source

    def get_dest_node(self):
        return self._dest

    def get_nodes(self) -> Tuple[Node, Node]:
        return self._source, self._dest

    def get_constraints(self) -> List[Constraint]:
        return [x for x in self._actions if isinstance(x, Constraint)]

    def get_updates(self) -> List[Update]:
        return [x for x in self._actions if isinstance(x, Update)]

    def get_actions(self):
        return self._actions

    def get_weight(self):
        return self._weight

    def get_channel(self) -> DirectedChannel:
        return self._channel

    def visit(self, visitor: Visitor):
        visitor.visit_node(self._dest)
        visitor.visit_expr(self._weight)
        for a in self._actions:
            if isinstance(a, Update):
                visitor.visit_update(a)
            elif isinstance(a, Constraint):
                visitor.visit_constraint(a)


class Automata(Visitable):
    def __init__(self, start_nodes: Node, nodes: Set[Node], edges: List[Edge], local_variables: Set[Variable]):
        self._start_node = start_nodes
        self._nodes = nodes
        self._edges = edges
        self._variables = local_variables

    def validate_internal_consistency(self):
        if self._start_node not in self._nodes:
            raise ModelException(f"Start node {self._start_node} not in set of nodes contained by automata")
        for e in self._edges:
            s, e = e.get_nodes()
            if s not in self._nodes:
                raise ModelException(f"edge mapping node '{s}' to '{e}' does not contain '{s}' as a node in automta")
            if e not in self._nodes:
                raise ModelException(f"edge mapping node '{s}' to '{e}' does not contain '{e}' as a node in automta")

    def _check_variable_overlap(self, other: Set[Variable]):
        n = self._variables.intersection(other)
        if len(n) > 0:
            raise ModelException(f"Variable {n} already exists in global scope")

    def get_start_node(self):
        return self._start_node

    def get_edges(self):
        return self._edges

    def get_all_nodes(self) -> Iterator:
        seen = set()
        for e in self._edges:
            s, d = e.get_nodes()
            if s not in seen:
                seen.add(s)
                yield s
            if d not in seen:
                seen.add(d)
                yield d

    def get_local_variables(self):
        return self._variables

    def visit(self, visitor: Visitor):
        self._start_node.visit(visitor)
        for v in self.get_local_variables():
            visitor.visit_variable(v)


class Network(Visitable):
    def __init__(self, automatas: List[Automata] = [], global_variables: Set[Variable] = set()):
        self._automatas = automatas
        self._variables = global_variables

    def get_automatas(self):
        return self._automatas

    def get_variables(self):
        return self._variables

    def add_automata(self, aut: Automata) -> Automata:
        self._automatas.append(aut)
        return aut

    def visit(self, visitor: Visitor):
        for v in self._variables:
            visitor.visit_variable(v)
        for a in self._automatas:
            visitor.visit_automata(a)
