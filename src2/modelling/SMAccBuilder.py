from SMAccVisitors import *
import tempfile
import os
import subprocess
import xml.etree.ElementTree as xml


class SMAccBuilder:
    def __init__(self, executable_path: str = None):
        self._ex_path = executable_path
        self._node_map: Dict[Node, int] = dict()
        self._channel_map: Dict[Channel | DirectedChannel, int] = dict()

    def get_id(self, n: Node):
        id = self._node_map[n] = self._node_map.get(n, len(self._node_map) + 1)
        return str(id)

    def get_channel_id(self, dc: DirectedChannel):
        if dc in self._channel_map:
            id_ = self._channel_map.get(dc)
        else:
            id_ = self._channel_map[dc] = len(self._channel_map)
        return id_ if dc.is_broadcast() else -id_

    def write_help(self):
        if self._ex_path is None:
            raise Exception("No executable path is given to SMAccBuilder object. Cannot build and run model.")
        subprocess.run([self._ex_path, "-h"])

    def build_and_run(self, network: Network, *,
                      epsilon=0.005, alpha=0.1, blocks=40, threads=256,
                      use_gpu=True, end_criteria='100t', use_jit=False, use_sm=False,
                      write_mode='c', silent=False, cpu_threads=1, output_name="output"):
        if self._ex_path is None:
            raise Exception("No executable path is given to SMAccBuilder object. Cannot build and run model.")
        self._channel_map.clear()
        self._channel_map[Channel(TAU_CHANNEL)] = 0
        file = tempfile.mktemp(suffix=".SMAcc")
        self.build(file, network)

        parameters = [self._ex_path, "-m", file, "-o", output_name, "-a", str(alpha), "-e", str(epsilon), "-c",
                      str(cpu_threads), "-w", write_mode, "-b", f"{blocks},{threads}", "-x", str(end_criteria), "-d"]

        if use_gpu:
            parameters.append('0')
        else:
            parameters.append('1')
        if use_jit:
            parameters.append("-j")

        if use_sm:
            parameters.append("-s")
        if silent:
            parameters.append("-v")
            parameters.append("1")

        print(' '.join(parameters))
        subprocess.run(parameters)
        # os.remove(file)

    def build(self, path: str, network: Network):
        self._node_map.clear()  # reset
        scope_v = SafetyVisitor()

        # throws error to user if not valid
        scope_v.visit_network(network)

        root = xml.Element("Network")

        for v in network.get_variables():
            self._write_variable(root, v)
        for a in network.get_automatas():
            self._write_automata(root, a)

        tree = xml.ElementTree(root)
        xml.indent(tree, space='\t', level=0)
        tree.write(path)

    def _write_automata(self, root: xml.Element, a: Automata):
        aroot = xml.Element("Automata")
        for v in a.get_local_variables():
            self._write_variable(aroot, v)
        self._write_node(aroot, a.get_start_node(), is_start_node=True)
        for n in a.get_all_nodes():
            if n == a.get_start_node(): continue
            self._write_node(aroot, n)
        for e in a.get_edges():
            self._write_edge(aroot, e)
        root.append(aroot)

    def _write_node(self, root: xml.Element, n: Node, *, is_start_node=False):
        nroot = xml.Element("Node")
        nroot.set("id", self.get_id(n))
        nroot.set("name", n.get_name())
        nroot.set("init", str(is_start_node))
        nroot.set("branch", str(n.is_branch_node()))
        nroot.set("lambda", n.get_rate_of_exponential().get_expr_str())
        for i in n.get_invariants():
            self._write_constraint(nroot, i)
        root.append(nroot)

    def _write_edge(self, root: xml.Element, e: Edge):
        eroot = xml.Element("Edge")

        eroot.set("weight", e.get_weight().get_expr_str())

        s, d = e.get_nodes()
        eroot.set("source_id", self.get_id(s))
        eroot.set("dest_id", self.get_id(d))

        ch = e.get_channel()
        ch_int = self.get_channel_id(ch)
        eroot.set("channel", str(ch_int))

        for c in e.get_constraints():
            self._write_constraint(eroot, c)
        for u in e.get_updates():
            self._write_update(eroot, u)
        root.append(eroot)

    def _write_constraint(self, root: xml.Element, i: Constraint):
        croot = xml.Element("Constraint")
        croot.set("type", i.get_type())
        lv, le = i.get_left()
        rv, re = i.get_right()
        if lv is not None:
            croot.set("left_var", lv.get_name())
        else:
            croot.set("left_expr", le.get_expr_str())
        if rv is not None:
            croot.set("right_var", rv.get_name())
        else:
            croot.set("right_expr", re.get_expr_str())
        root.append(croot)

    def _write_update(self, root: xml.Element, u: Update):
        att = {
            "var": u.get_variable().get_name(),
            "expr": u.get_expr().get_expr_str()
        }
        xml.SubElement(root, "Update", att)

    def _write_variable(self, root: xml.Element, v: Variable):
        att = {
            "name": v.get_name(),
            "value": str(float(v.get_init_value())),
            "rate": str(v.get_rate()),
            "track": str(v.should_track())
        }
        xml.SubElement(root, "Variable", att)
