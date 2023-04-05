from typing import Dict, Set, List, Tuple
from SMAccDomain import *


class Visitor:
    def __init__(self):
        self._seen = set()

    def visit_network(self, ne: Network):
        ne.visit(self)

    def visit_automata(self, a: Automata):
        a.visit(self)

    def visit_node(self, n: Node):
        if n in self._seen: return
        n.visit(self)

    def visit_edge(self, e: Edge):
        e.visit(self)

    def visit_variable(self, v: Variable):
        v.visit(self)

    def visit_constraint(self, c: Constraint):
        c.visit(self)

    def visit_update(self, u: Update):
        u.visit(self)

    def visit_expr(self, ex: Expr):
        ex.visit(self)

    def visit_channel(self, ch: Channel):
        ch.visit(self)


class SafetyVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.global_scope = None
        self.local_scope = None

    def contains_var(self, var):
        return var in self.global_scope or var in self.local_scope

    @staticmethod
    def check_scope_overlap(scope1: Set[Variable], scope2: Set[Variable]):
        overlap = scope1.intersection(scope2)
        if len(overlap) > 0:
            raise ModelException(f"Variable(s) {overlap} overlaps in scope")

    def visit_network(self, ne: Network):
        self.global_scope = ne._variables
        ne.visit(self)

    def visit_automata(self, a: Automata):
        a.validate_internal_consistency()
        SafetyVisitor.check_scope_overlap(self.global_scope, a.get_local_variables())
        self.local_scope = a.get_local_variables()
        a.visit(self)

    def visit_variable(self, v: Variable):
        if not self.contains_var(v):
            raise ModelException(f"variable {v.get_name()} is not in local or global scope")

    def visit_constraint(self, c: Constraint):
        if not c.validate(self.global_scope, self.local_scope):
            raise ModelException(f"Constraint {c} contains clocks in expression, which is not valid")
        c.visit(self)

    def visit_expr(self, ex: Expr):
        for var in ex.get_variables_ref():
            if not self.contains_var(var):
                raise ModelException(
                    f"variable {var} used in expression {ex} not in scope\n global: {self.global_scope}\n local: {self.local_scope} ")

    def visit_channel(self, ch: Channel):
        if not isinstance(ch, DirectedChannel) and isinstance(ch, Channel):
            raise ModelException("Channel on edge is not directed. Should be either a listener or broadcast channel")
