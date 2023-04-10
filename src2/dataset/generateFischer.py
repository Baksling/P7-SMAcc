N = 10
K = 0

result = ""

declaration = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>
<nta>
	<declaration>// Fischer's mutual exclusion protocol.
int id;</declaration>"""

result += declaration

number_of_nodes = 4
template = ""
for current_i in range(N):
    offset = current_i * number_of_nodes
    template += f"""<template>
		<name x="16" y="-8">P{current_i}</name>
		<declaration>clock x;
const int k = 2;
int pid = {current_i};</declaration>
		<location id="id{offset+0}" x="216" y="176">
			<name x="216" y="192">wait</name>
			<label kind="exponentialrate" x="232" y="168">1</label>
		</location>
		<location id="id{offset+1}" x="216" y="48">
			<name x="216" y="16">req</name>
			<label kind="invariant" x="240" y="32">x&lt;=k</label>
		</location>
		<location id="id{offset+2}" x="64" y="48">
			<name x="54" y="18">A</name>
			<label kind="exponentialrate" x="40" y="40">1</label>
		</location>
		<location id="id{offset+3}" x="64" y="176">
			<name x="56" y="192">cs</name>
			<label kind="exponentialrate" x="40" y="168">1</label>
		</location>
		<init ref="id{offset+2}"/>
		<transition>
			<source ref="id{offset+2}"/>
			<target ref="id{offset+1}"/>
			<label kind="guard" x="88" y="24">id== 0</label>
			<label kind="assignment" x="160" y="24">x = 0</label>
		</transition>
		<transition>
			<source ref="id{offset+1}"/>
			<target ref="id{offset+0}"/>
			<label kind="guard" x="144" y="72">x&lt;=k</label>
			<label kind="assignment" x="144" y="104">x = 0,
id = pid</label>
		</transition>
		<transition>
			<source ref="id{offset+0}"/>
			<target ref="id{offset+1}"/>
			<label kind="guard" x="264" y="120">id== 0</label>
			<label kind="assignment" x="264" y="88">x = 0</label>
			<nail x="251" y="146"/>
			<nail x="251" y="82"/>
		</transition>
		<transition>
			<source ref="id{offset+0}"/>
			<target ref="id{offset+3}"/>
			<label kind="guard" x="96" y="184">x&gt;k &amp;&amp; id==pid</label>
		</transition>
		<transition>
			<source ref="id{offset+3}"/>
			<target ref="id{offset+2}"/>
			<label kind="assignment" x="8" y="80">id = 0</label>
		</transition>
	</template>"""

result += template

system = """<system>system """
for current_i in range(N):
    system += f'P{current_i},' if current_i < N - 1 else f'P{current_i};</system>'

result += system

query = """<queries>
		<query>
			<formula>Pr[ &lt;= 300](&lt;&gt; P(1).cs)
			</formula>
			<comment>Fischer's mutual exclusion protocol.
			</comment>
		</query>
		<query>
			<formula>A[] forall (i:id_t) forall (j:id_t) P(i).cs &amp;&amp; P(j).cs imply i == j
			</formula>
			<comment>Mutex requirement.
			</comment>
		</query>
		<query>
			<formula>A[] not deadlock
			</formula>
			<comment>The system is deadlock free.
			</comment>
		</query>
		<query>
			<formula>P(1).req --&gt; P(1).wait
			</formula>
			<comment>Whenever P(1) requests access to the critical section it will eventually enter the wait state.
			</comment>
		</query>
	</queries>
</nta>"""

result += query

filename = f'fischer_{N}_{K}.xml'
with open(filename, 'w') as f:
    f.write(result)
