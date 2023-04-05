from StochasticModel import *

TRADER_COUNT = 10
EAGER_DEGREE = 0.1
STARTING_CAPITAL = 100
TOTAL_SHARES = 1000
MARKET_START_PRICE = 100
SHARE_COUNT_NAME = "share_count"
SHARE_PRICE_NAME = "share_price"
MARKET_FREQUENCY = 10
MARKET_VARIANCE = 10


class FinancialModel(StochasticModel):

    @staticmethod
    def build_market(shares_count: Variable, share_price: Variable) -> Automata:
        market = Node("market", exponential_rate=str(MARKET_FREQUENCY))
        oob_check = Node("oob_check", is_branch_node=True)

        increase = Edge(market, market, actions={
            Update(share_price, f"{share_price} + random({MARKET_VARIANCE})")
        }, weight=f"{TOTAL_SHARES} - {shares_count}")
        decrease = Edge(market, oob_check, actions={
            Update(share_price, f"{share_price} + random({MARKET_VARIANCE})"),
        }, weight=f"{shares_count}")
        check_oob = Edge(oob_check, market, actions={
            Update(share_price, ConditionalExpr(f"{share_price} <= 1", "1", f"{share_price}"))
        })

        return Automata(market,
                        nodes={market, oob_check},
                        edges=[increase, decrease, check_oob],
                        local_variables=set())

    @staticmethod
    def build_trader(index: int, share_count: Variable, share_price: Variable) -> Automata:
        my_share_count = Variable(f"share_count_{index}", init_value=0, rate=0, track_value=True)
        total_money = Variable(f"wallet_{index}", init_value=STARTING_CAPITAL, rate=0, track_value=True)
        last_seen_price = Variable(f"last_price_{index}", init_value=MARKET_START_PRICE, rate=0, track_value=False)
        roe = ConditionalExpr(f"{share_price} > {last_seen_price} ", 
                              f"{EAGER_DEGREE} * ({share_price} - {last_seen_price})", 
                              f"{EAGER_DEGREE} * ({last_seen_price} - {share_price})")

        location = Node(f"trader_{index}", exponential_rate=roe)

        buy = Edge(location, location,
                   actions={
                       Constraint(share_count, '>', "0"),
                       Constraint(share_price, '<', total_money),
                       Update(share_count, f"{share_count} - 1"),
                       Update(my_share_count, f"{my_share_count} + 1"),
                       Update(last_seen_price, share_price.get_name()),
                       Update(total_money, f"{total_money} - {share_price}")
                   })

        sell = Edge(location, location,
                    actions={
                        Constraint(my_share_count, '>', '0'),
                        Update(share_count, f"{share_count} + 1"),
                        Update(my_share_count, f"{my_share_count} - 1"),
                        Update(last_seen_price, share_price.get_name()),
                        Update(total_money, f"{total_money.get_name()} + {share_price}")
                    })
        return Automata(
            start_nodes=location,
            nodes={location},
            edges=[buy, sell],
            local_variables={my_share_count, total_money, last_seen_price})

    def setup_model(self) -> Network:
        automata = []
        
        shares = Variable(SHARE_COUNT_NAME, TOTAL_SHARES, rate=0, track_value=True)
        prices = Variable(SHARE_PRICE_NAME, MARKET_START_PRICE, rate=0, track_value=True)
        automata.append(FinancialModel.build_market(shares, prices))

        for i in range(TRADER_COUNT):
            automata.append(FinancialModel.build_trader(i, shares, prices))

        return Network(automatas=automata, global_variables={shares, prices})


if __name__ == "__main__":
    model = FinancialModel()
    model.run_cli()
