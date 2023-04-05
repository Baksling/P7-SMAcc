N = 1000

result = ""

declartion = \
f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>
<nta>
	<declaration>
const int N={N};
broadcast chan busy;
const int macMaxCSMABackoffs = 4;
const int UnitBackoff = 20;
const int CCA_duration = 8;
const int CCA = 8;
const int aTurnaroundTime = 12;
const int TurnAround = 12;
const int ACK_time = 88;
const int ACK_Time = 88;
const int ACK = 88;
const int MaxFrameRetries = 3;
const int macMinBE = 3;
const int MaxBE = 5;
const int MaxNB = 5;
const int FrameLength = 35*8;
const int MaxWakingDelay=1000;
const int MinLIFS = 40;
const bool is_discrete_waiting = false;
const bool acknowledgment_supported = true;
const bool recover_from_failures = false;
const int wait_after_failure = 0;</declaration>"""

result += declartion

template_id = 0
increment = 18
for current_i in range(0,N*increment, increment):
    template = \
    f"""	<template>
            <name x="5" y="5">Process{template_id}</name>
            <declaration>
    int be;
    int nb; 
    int nt;
    int nretries;
    int backoff;
    int waking_delay;
    clock x;
    bool cca_passed := false;
    bool collision_occured := false;
    </declaration>
            <location id="id{current_i+0}" x="-2352" y="64">
                <label kind="invariant" x="-2346" y="34">x&lt;=ACK_Time</label>
            </location>
            <location id="id{current_i+1}" x="-2560" y="-200">
                <urgent/>
            </location>
            <location id="id{current_i+2}" x="-2264" y="-200">
                <name x="-2264" y="-184">FAILURE</name>
            </location>
            <location id="id{current_i+3}" x="-1928" y="-200">
                <urgent/>
            </location>
            <location id="id{current_i+4}" x="-2704" y="-288">
                <label kind="invariant" x="-2712" y="-328">x&lt;=MaxWakingDelay</label>
            </location>
            <location id="id{current_i+5}" x="-2560" y="-40">
                <label kind="invariant" x="-2544" y="-72">x&lt;=MinLIFS</label>
            </location>
            <location id="id{current_i+6}" x="-2560" y="-288">
                <urgent/>
            </location>
            <location id="id{current_i+7}" x="-2336" y="-128">
                <name x="-2416" y="-152">SUCCESS</name>
            </location>
            <location id="id{current_i+8}" x="-2336" y="-40">
                <urgent/>
            </location>
            <location id="id{current_i+9}" x="-2264" y="-360">
                <urgent/>
            </location>
            <location id="id{current_i+10}" x="-1928" y="64">
                <label kind="invariant" x="-1920" y="32">x&lt;=TurnAround</label>
            </location>
            <location id="id{current_i+11}" x="-2160" y="-40">
                <label kind="invariant" x="-2152" y="-72">x&lt;=ACK_time</label>
            </location>
            <location id="id{current_i+12}" x="-2160" y="-200">
                <label kind="invariant" x="-2232" y="-232">x&lt;=aTurnaroundTime</label>
            </location>
            <location id="id{current_i+13}" x="-1928" y="-360">
                <name x="-2064" y="-392">TRANSMIT_DATA</name>
                <label kind="invariant" x="-2064" y="-376">x&lt;=FrameLength</label>
            </location>
            <location id="id{current_i+14}" x="-1928" y="-488">
                <name x="-2032" y="-512">VULNERABLE</name>
                <label kind="invariant" x="-1912" y="-512">x&lt;=TurnAround</label>
            </location>
            <location id="id{current_i+15}" x="-2264" y="-488">
                <name x="-2312" y="-512">CCA</name>
                <label kind="invariant" x="-2256" y="-480">x&lt;=CCA</label>
            </location>
            <location id="id{current_i+16}" x="-2560" y="-488">
                <name x="-2680" y="-512">WAIT_BACKOFF</name>
                <label kind="invariant" x="-2744" y="-488">x&lt;=backoff*UnitBackoff</label>
            </location>
            <location id="id{current_i+17}" x="-2560" y="-360">
                <urgent/>
            </location>
            <init ref="id{current_i+4}"/>
            <transition>
                <source ref="id{current_i+0}"/>
                <target ref="id{current_i+5}"/>
                <label kind="guard" x="-2488" y="40">x==ACK_Time</label>
                <label kind="assignment" x="-2542" y="64">x:=0</label>
                <nail x="-2560" y="64"/>
            </transition>
            <transition>
                <source ref="id{current_i+1}"/>
                <target ref="id{current_i+2}"/>
                <label kind="guard" x="-2536" y="-200">nretries == (MaxFrameRetries-1)</label>
                <nail x="-2360" y="-200"/>
            </transition>
            <transition>
                <source ref="id{current_i+1}"/>
                <target ref="id{current_i+6}"/>
                <label kind="guard" x="-2552" y="-240">nretries &lt; (MaxFrameRetries-1)</label>
                <label kind="assignment" x="-2552" y="-224">nretries = nretries + 1</label>
            </transition>
            <transition>
                <source ref="id{current_i+4}"/>
                <target ref="id{current_i+6}"/>
            </transition>
            <transition>
                <source ref="id{current_i+5}"/>
                <target ref="id{current_i+1}"/>
                <label kind="guard" x="-2552" y="-176">x==MinLIFS</label>
            </transition>
            <transition>
                <source ref="id{current_i+6}"/>
                <target ref="id{current_i+17}"/>
                <label kind="assignment" x="-2552" y="-336">be:=macMinBE, nb:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+15}"/>
                <target ref="id{current_i+15}"/>
                <label kind="synchronisation" x="-2304" y="-592">busy?</label>
                <label kind="assignment" x="-2304" y="-576">cca_passed:=false</label>
                <nail x="-2304" y="-552"/>
                <nail x="-2216" y="-552"/>
            </transition>
            <transition>
                <source ref="id{current_i+11}"/>
                <target ref="id{current_i+8}"/>
                <label kind="guard" x="-2304" y="-72">x==ACK</label>
                <label kind="assignment" x="-2304" y="-56">nt:=nt-1</label>
            </transition>
            <transition>
                <source ref="id{current_i+9}"/>
                <target ref="id{current_i+2}"/>
                <label kind="guard" x="-2256" y="-328">nb == MaxNB</label>
            </transition>
            <transition>
                <source ref="id{current_i+9}"/>
                <target ref="id{current_i+17}"/>
                <label kind="guard" x="-2544" y="-400">nb &lt; MaxNB</label>
                <label kind="assignment" x="-2544" y="-384">be:= be+1 &gt; MaxBE ? MaxBE: be+1</label>
            </transition>
            <transition>
                <source ref="id{current_i+11}"/>
                <target ref="id{current_i+11}"/>
                <label kind="synchronisation" x="-2144" y="-32">busy?</label>
                <label kind="assignment" x="-2144" y="-16">collision_occured:=true</label>
                <nail x="-2184" y="8"/>
                <nail x="-2136" y="8"/>
                <nail x="-2152" y="-16"/>
            </transition>
            <transition>
                <source ref="id{current_i+3}"/>
                <target ref="id{current_i+12}"/>
                <label kind="guard" x="-2080" y="-224">collision_occured == false</label>
            </transition>
            <transition>
                <source ref="id{current_i+10}"/>
                <target ref="id{current_i+0}"/>
                <label kind="guard" x="-2056" y="32">x == TurnAround</label>
                <label kind="assignment" x="-2056" y="48">x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+3}"/>
                <target ref="id{current_i+10}"/>
                <label kind="guard" x="-1920" y="-176">collision_occured == true</label>
                <nail x="-1928" y="-64"/>
            </transition>
            <transition>
                <source ref="id{current_i+13}"/>
                <target ref="id{current_i+13}"/>
                <label kind="synchronisation" x="-1864" y="-384">busy?</label>
                <label kind="assignment" x="-1864" y="-368">collision_occured:=true</label>
                <nail x="-1872" y="-392"/>
                <nail x="-1872" y="-320"/>
            </transition>
            <transition>
                <source ref="id{current_i+8}"/>
                <target ref="id{current_i+7}"/>
                <label kind="guard" x="-2328" y="-96">collision_occured == false</label>
                <nail x="-2336" y="-72"/>
            </transition>
            <transition>
                <source ref="id{current_i+8}"/>
                <target ref="id{current_i+5}"/>
                <label kind="guard" x="-2480" y="-32">collision_occured == true</label>
                <label kind="assignment" x="-2408" y="-64">x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+12}"/>
                <target ref="id{current_i+11}"/>
                <label kind="guard" x="-2152" y="-176">x==TurnAround</label>
                <label kind="synchronisation" x="-2152" y="-120">busy!</label>
                <label kind="assignment" x="-2152" y="-160">collision_occured:=nt&gt;0,
    nt:=nt+1,
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+13}"/>
                <target ref="id{current_i+3}"/>
                <label kind="guard" x="-1920" y="-312">x==FrameLength</label>
                <label kind="assignment" x="-1920" y="-296">x:=0,
    nt:=nt-1</label>
            </transition>
            <transition>
                <source ref="id{current_i+14}"/>
                <target ref="id{current_i+13}"/>
                <label kind="guard" x="-1920" y="-472">x==TurnAround</label>
                <label kind="synchronisation" x="-2184" y="-520">busy!</label>
                <label kind="assignment" x="-1920" y="-456">x:=0,
    collision_occured:= nt&gt;0,
    nt:=nt+1</label>
            </transition>
            <transition>
                <source ref="id{current_i+15}"/>
                <target ref="id{current_i+14}"/>
                <label kind="guard" x="-2184" y="-536">x==CCA_duration &amp;&amp; cca_passed == true</label>
                <label kind="assignment" x="-2184" y="-504">x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+15}"/>
                <target ref="id{current_i+9}"/>
                <label kind="guard" x="-2264" y="-448">x == CCA_duration &amp;&amp; cca_passed == false</label>
                <label kind="assignment" x="-2264" y="-432">nb := nb+1</label>
            </transition>
            <transition>
                <source ref="id{current_i+16}"/>
                <target ref="id{current_i+15}"/>
                <label kind="guard" x="-2512" y="-544">x==backoff*UnitBackoff</label>
                <label kind="assignment" x="-2512" y="-528">cca_passed:= nt==0,
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 0 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 1 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 2 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 3 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 4 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 5 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 6 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 7 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 8 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 9 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 10 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 11 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 12 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 13 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 14 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 15 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 16 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 17 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 18 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 19 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 20 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 21 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 22 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 23 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 24 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 25 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 26 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 27 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 28 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 29 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 30 % (2^be),
    x:=0</label>
            </transition>
            <transition>
                <source ref="id{current_i+17}"/>
                <target ref="id{current_i+16}"/>
                <label kind="assignment" x="-2552" y="-448">backoff := 31 % (2^be),
    x:=0</label>
            </transition>
        </template>"""
    
    template_id += 1
    result += template

system = """<system>
system """

for i in range(N):
    system += f"""Process{i},""" if i < N - 1 else f"""Process{i};"""

system += "</system>"

result += system

e = """<queries>
		<query>
			<formula>Pr[&lt;=100](&lt;&gt;sum(i:id_t) Process0(i).SUCCESS)</formula>
			<comment></comment>
		</query>
	</queries>
</nta>"""

result += e

filename = f'CSMA_{N}.xml'
with open(filename, 'w') as f:
    f.write(result)