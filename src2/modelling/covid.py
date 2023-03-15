from SMAccDomain import *
from StochasticModel import StochasticModel


class CovidModel(StochasticModel):
    def __init__(self):
        super(CovidModel, self).__init__()

    def setup_model(self) -> Network:
        N = 10000
        BRN = 2.4
        GAMMA = 1.0 / 3.4
        BETA0 = BRN * GAMMA
        ALPHA = 1.0 / 5.1
        PH = 9.0 / 10000
        KAPPA = (GAMMA * PH) / (1.0 - PH)
        TAU = 1.0 / 10.12

        s = Variable("S", 9900, rate=0)
        e = Variable("E", 100, rate=0)
        i = Variable("I", 0, rate=0)
        h = Variable("H", 0, rate=0)
        r = Variable("R", 0, rate=0)
        g_vars = [s, e, i, h, r]

        n1 = Node("n1", exponential_rate=f"{BETA0}*S*I/{N}")
        n2 = Node("n2", exponential_rate=f"{ALPHA}*E")
        n3 = Node('n3', exponential_rate=f"{KAPPA}*I")
        n4 = Node('n4', exponential_rate=f"{TAU}*H")
        n5 = Node('n5', exponential_rate=f"{GAMMA}*I")

        e1 = Edge(n1, n1, actions={
            Constraint(s, '>', '0'),
            Constraint(i, '>', '0'),
            Update(s, Expr("S-1")),
            Update(e, Expr("E+1"))
        })

        e2 = Edge(n2, n2, actions={
            Constraint(e, '>', '0'),
            Update(e, "E-1"),
            Update(i, "I+1")
        })

        e3 = Edge(n3, n3, actions={
            Constraint(i, '>', '0'),
            Update(i, "I-1"),
            Update(h, "H+1")
        })

        e4 = Edge(n4, n4,
                  actions={
                      Constraint(h, '>', '0'),
                      Update(h, "H-1"),
                      Update(r, "R+1")
                  })

        e5 = Edge(n5, n5, actions={
            Constraint(i, '>', '0'),
            Update(i, "I-1"),
            Update(r, "R+1")
        })

        network = Network(global_variables=set(g_vars))
        for n, e in [(n1, e1), (n2, e2), (n3, e3), (n4, e4), (n5, e5)]:
            network.add_automata(Automata(n, nodes={n}, edges=[e], local_variables=set()))

        return network


if __name__ == "__main__":
    model = CovidModel()
    model.run_cli()
