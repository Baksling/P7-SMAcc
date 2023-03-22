
n = 10000
s = '<?xml version="1.0" encoding="utf-8"?> \n<!DOCTYPE nta PUBLIC "-//Uppaal Team//DTD Flat System 1.1//EN" "http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd">\n<nta>\n<declaration>const int N=10000; \n const double BRN = 2.4; \n const double alpha = 1.0/5.1; \n const double gamma = 1.0/3.4; \n const double beta = BRN * gamma; \n const double pH = 9.0/10000.0; \n const double kappa = gamma * pH / (1.0-pH); \n const double tau = 1.0/10.12; \n int inf = 0;\n</declaration>\n'
name = ""
nodeId = 0
persons = ""

e = '\n<queries>\n<query>\n<formula>simulate[&lt;=100; 1]{S,E,I,H,R}\n</formula>\n<comment>\n</comment>\n</query>\n</queries>\n</nta>'

with open("test.xml", "w") as f:
    f.write(s)
    for i in range(0, n):
        name = "person" + str(i)
        persons += name if i == n-1 else name + ", "
        temp = f'<template>\n<name x="5" y="5">{name}\n</name>\n<declaration>// Place local declarations here.\n</declaration>\n<location id="id{nodeId}" x="-144" y="-161">\n<name x="-154" y="-195">S\n</name>\n<label kind="exponentialrate" x="-154" y="-127">beta*inf/N\n</label>\n</location>\n<location id="id{nodeId+1}" x="-42" y="-161">\n<name x="-52" y="-195">E\n</name>\n<label kind="exponentialrate" x="-52" y="-127">alpha\n</label>\n</location>\n<location id="id{nodeId+2}" x="59" y="-161">\n<name x="49" y="-195">I\n</name>\n<label kind="exponentialrate" x="17" y="-127">kappa+gamma\n</label>\n</location>\n<location id="id{nodeId+3}" x="212" y="-161">\n<name x="202" y="-195">R\n</name>\n</location>\n<location id="id{nodeId+4}" x="119" y="-51">\n<name x="93" y="-59">H\n</name>\n<label kind="exponentialrate" x="110" y="-34">tau\n</label>\n</location>\n<branchpoint id="id{nodeId+5}" x="119" y="-161">\n</branchpoint>\n<init ref="id{nodeId}"/>\n<transition>\n<source ref="id{nodeId+4}"/>\n<target ref="id{nodeId+3}"/>\n</transition>\n<transition>\n<source ref="id{nodeId+5}"/>\n<target ref="id{nodeId+4}"/>\n<label kind="assignment" x="119" y="-106">inf = inf - 1\n</label>\n<label kind="probability" x="119" y="-89">kappa\n</label>\n</transition>\n<transition>\n<source ref="id{nodeId+5}"/>\n<target ref="id{nodeId+3}"/>\n<label kind="assignment" x="137" y="-157">inf = inf - 1\n</label>\n<label kind="probability" x="137" y="-140">gamma\n</label>\n</transition>\n<transition>\n<source ref="id{nodeId+2}"/>\n<target ref="id{nodeId+5}"/>\n</transition>\n<transition>\n<source ref="id{nodeId+1}"/>\n<target ref="id{nodeId+2}"/>\n<label kind="assignment" x="-24" y="-161">inf = inf + 1\n</label>\n</transition>\n<transition>\n<source ref="id{nodeId}"/>\n<target ref="id{nodeId+1}"/>\n</transition>\n</template>'
        f.write(temp)
        nodeId += 6
    system = f"\n<system>system {persons};\n</system>"
    f.write(system)
    f.write(e)