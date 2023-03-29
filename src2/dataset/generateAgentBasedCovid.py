number_of_automatas = 100000
n = 1
infected_percent = number_of_automatas*0.01
kappa = 265
gamma = 294_117

s = f'<?xml version="1.0" encoding="utf-8"?> \n<!DOCTYPE nta PUBLIC "-//Uppaal Team//DTD Flat System 1.1//En" "http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd">\n<nta>\n<declaration>int p = {n}; \nconst double brn = 2.4; \nconst double alpha = 1.0/5.1; \nconst double gamma = 1.0/3.4; \nconst double beta = brn * gamma; \nconst double pH = 9.0/10000.0; \nconst double kappa = gamma * pH / (1.0-pH); \nconst double tau = 1.0/10.12; \nint inf = {int(infected_percent)};\n</declaration>\n'
name = ""
nodeId = 0
persons = ""
do_da_old = False

e = '\n<queries>\n<query>\n<formula>simulate[&lt;=100; 1]{inf}\n</formula>\n<comment>\n</comment>\n</query>\n</queries>\n</nta>'

with open(f"agentBaseCovid_{number_of_automatas}_{infected_percent}.xml", "w") as f:
    f.write(s)
    for i in range(0, n):
        name = "person" + str(i)
        persons += name if i == n-1 else name + ", "
        temp = f'<template>\n<name x="5" y="5">{name}\n</name>\n<declaration>// Place local declarations here.\n</declaration>\n<location id="id{nodeId}" x="-144" y="-161">\n<name x="-154" y="-195">S</name>\n<label kind="exponentialrate" x="-154" y="-127">beta * inf / p</label>\n</location>\n<location id="id{nodeId+1}" x="-42" y="-161">\n<name x="-52" y="-195">E</name>\n<label kind="exponentialrate" x="-52" y="-127">alpha</label>\n</location>\n<location id="id{nodeId+2}" x="59" y="-161">\n<name x="49" y="-195">I\n</name>\n<label kind="exponentialrate" x="17" y="-127">kappa+gamma</label>\n</location>\n<location id="id{nodeId+3}" x="212" y="-161">\n<name x="202" y="-195">R\n</name>\n</location>\n<location id="id{nodeId+4}" x="119" y="-51">\n<name x="93" y="-59">H\n</name>\n<label kind="exponentialrate" x="110" y="-34">tau</label>\n</location>\n<branchpoint id="id{nodeId+5}" x="119" y="-161">\n</branchpoint>\n<init ref="id{nodeId}"/>\n<transition>\n<source ref="id{nodeId+4}"/>\n<target ref="id{nodeId+3}"/>\n</transition>\n<transition>\n<source ref="id{nodeId+5}"/>\n<target ref="id{nodeId+4}"/>\n<label kind="assignment" x="119" y="-106">inf = inf - 1</label>\n<label kind="probability" x="119" y="-89">{kappa}\n</label>\n</transition>\n<transition>\n<source ref="id{nodeId+5}"/>\n<target ref="id{nodeId+3}"/>\n<label kind="assignment" x="137" y="-157">inf = inf - 1</label>\n<label kind="probability" x="137" y="-140">{gamma}\n</label>\n</transition>\n<transition>\n<source ref="id{nodeId+2}"/>\n<target ref="id{nodeId+5}"/>\n</transition>\n<transition>\n<source ref="id{nodeId+1}"/>\n<target ref="id{nodeId+2}"/>\n<label kind="assignment" x="-24" y="-161">inf = inf + 1</label>\n</transition>\n<transition>\n<source ref="id{nodeId}"/>\n<target ref="id{nodeId+1}"/>\n</transition>\n</template>'
        f.write(temp)
        nodeId += 6
    system = f"\n<system>system {persons};\n</system>"
    f.write(system)
    f.write(e)
    