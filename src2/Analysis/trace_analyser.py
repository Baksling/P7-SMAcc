import sqlite3
import sys
import tempfile
import argparse
import os
import os.path as path
import csv


def setup_db(cache, source_folder, name):
    db = sqlite3.connect(cache)
    try:
        # setup tables
        db.execute("""
        CREATE TABLE nodes (
            node_id int PRIMARY KEY NOT NULL,
            subsystem int NOT NULL,
            name varchar(255),
            is_goal boolean NOT NULL
         );""")
        db.execute("""
         CREATE TABLE node_trace (
            sim_id int NOT NULL,
            step int NOT NULL,
            node_id int NOT NULL references nodes(node_id),
            time real NOT NULL,
            primary key (sim_id, step, node_id)
         );""")
        db.execute("""
         CREATE TABLE variable_trace(
            sim_id int NOT NULL,
            step int NOT NULL,
            variable_id int NOT NULL,
            value real NOT NULL,
            primary key (sim_id, step, variable_id)
         );
         """)

        # insert nodes
        node_records = []
        with open(path.join(source_folder, name + "_model.csv"), 'r') as f:
            for row in csv.DictReader(f, delimiter=','):
                node_records.append((row["node_id"], row["subsystem_id"], row["name"], row["is_goal"]))

        db.executemany("""INSERT INTO nodes VALUES(?,?,?,?);""", node_records)

        trace_records = []
        with open(path.join(source_folder, name + "_trace_nodes.csv"), 'r') as f:
            for row in csv.DictReader(f, delimiter=','):
                trace_records.append((row["sim_id"], row["step"], row["node_id"], row["time"]))

        db.executemany("""INSERT INTO node_trace VALUES(?,?,?,?);""", trace_records)

        variable_trace_records = []
        with open(path.join(source_folder, name + "_trace_variables.csv"), 'r') as f:
            for row in csv.DictReader(f, delimiter=','):
                variable_trace_records.append((row["sim_id"], row["step"], row["var_id"], row["value"]))

        db.executemany("""INSERT INTO variable_trace VALUES(?,?,?,?);""", variable_trace_records)

        db.commit()
    except Exception as ex:
        db.close()
        raise ex
    return db


def print_schema():
    print("""
        'nodes' table (The nodes in the system):
        node_id int PRIMARY KEY NOT NULL,
        subsystem int NOT NULL,
        name varchar(255),
        is_goal boolean NOT NULL
    """)
    print("""
        'node_trace' table (The trace of node traversals):
        sim_id int NOT NULL,
        step int NOT NULL,
        node_id int NOT NULL references NODES(node_id),
        time real NOT NULL,
        primary key (sim_id, step, node_id)
    """)
    print("""
        'variable_trace' table (The trace of variables from each step)
        sim_id int NOT NULL,
        step int NOT NULL,
        variable_id int NOT NULL,
        value real NOT NULL,
        primary key (sim_id, step, variable_id)
    """)
    return


def print_help():
    print("Write 'show schema' to show sql schema")
    print("Write 'exit' to exit.")
    print("Write 'q ' followed by named query, for pre-written queries.")
    print("Prewritten queries: \n\t" + ',\n\t'.join(['max time', 'hit goal']))


def execute_query(db, query):
    try:
        cursor = db.execute(query)
        for row in cursor:
            print(row)
    except sqlite3.OperationalError as ex:
        print(ex)


def common_queries(db, req):
    if "max time" in req:
        execute_query(db, "SELECT max(trace.step), trace.time FROM node_trace trace")
    elif "hit goal" in req:
        execute_query(db,
                      "SELECT sim_id, max(step), nodes.is_goal FROM node_trace trace JOIN nodes ON trace.node_id == nodes.node_id group by trace.sim_id")
    else:
        print("No query found")


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-s", "--source", help="source folder of data", dest="source")
    arg_parser.add_argument("-n", "--name", help="file name to loop for in data folder", dest="name")

    if len(sys.argv) <= 1:
        arg_parser.print_help()
        sys.exit(1)

    return arg_parser.parse_args()


def main(cache_file):
    args = parse_args()
    db = setup_db(cache_file, args.source, args.name)
    try:
        print("Hello, welcome to the trace analyser. Use 'help' to print help.")
        while True:
            print("\ninput: ")
            user_str: str = input()
            if "help" in user_str:
                print_help()
            elif "show schema" in user_str:
                print_schema()
            elif user_str.startswith("q "):
                common_queries(db, user_str)
            elif "exit" in user_str:
                print("Bye bye!")
                break
            else:
                execute_query(db, user_str)
    finally:
        db.close()


if __name__ == "__main__":
    cache = tempfile.mktemp(suffix=".sqlite")
    print(cache)
    try:
        main(cache)
    finally:
        os.remove(cache)
