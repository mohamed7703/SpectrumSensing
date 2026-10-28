"""CRN Network Simulation Module.

Provides discrete-event simulation of Cognitive Radio Networks with
topology modeling, wireless channel propagation, MAC protocol engines,
traffic generation, and performance trace analysis.
"""

from src.ns3_simulation.sim_engine import SimulationEngine, Event
from src.ns3_simulation.topology_builder import (
    Node,
    PrimaryUser,
    SecondaryUser,
    TopologyBuilder,
)
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.traffic_generator import TrafficGenerator, Packet
from src.ns3_simulation.mac_pure_aloha import PureAlohaMAC
from src.ns3_simulation.mac_slotted_aloha import SlottedAlohaMAC
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC
from src.ns3_simulation.trace_parser import TraceParser, PacketRecord
from src.ns3_simulation.nash_equilibrium import SpectrumContentionGame

__all__ = [
    "SimulationEngine",
    "Event",
    "Node",
    "PrimaryUser",
    "SecondaryUser",
    "TopologyBuilder",
    "ChannelModel",
    "TrafficGenerator",
    "Packet",
    "PureAlohaMAC",
    "SlottedAlohaMAC",
    "ILaCsmaMAC",
    "TraceParser",
    "PacketRecord",
    "SpectrumContentionGame",
]
