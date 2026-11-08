"""Record narrow, reproducible corrected mock/theory/timing measurements."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy import stats
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.math_models.energy_detection import calibrate_energy_threshold, compute_energy_statistic, compute_theoretical_threshold
from src.math_models.detection_metrics import binomial_rate_interval
from src.math_models.noise_models import compute_sample_complexity
from src.sdr_interface.mock_sdr import MockSDR
from src.ns3_simulation.sim_engine import SimulationEngine
from src.ns3_simulation.channel_model import ChannelModel
from src.ns3_simulation.topology_builder import PrimaryUser, SecondaryUser
from src.ns3_simulation.traffic_generator import Packet
from src.ns3_simulation.mac_ila_csma import ILaCsmaMAC
from src.ns3_simulation.mac_pure_aloha import PureAlohaMAC
from src.ns3_simulation.mac_slotted_aloha import SlottedAlohaMAC


def capture(config: dict, kind: str, seed: int, frames: int) -> np.ndarray:
    """Capture the specified number of frames through the configured mock ADC."""
    length = config["frame_length"]
    chunk = config["acquisition_chunk"]
    if chunk % length or frames * length % chunk:
        raise ValueError("frame counts and acquisition chunks must align exactly")
    receiver = MockSDR(signal_type=kind, random_seed=seed, snr_db=config["snr_db"],
                       sample_rate=config["sample_rate"], adc_bits=config["adc_bits"], gain=config["gain"])
    with receiver:
        return np.concatenate([receiver.read_samples(chunk).reshape(-1, length)
                               for _ in range(frames * length // chunk)])


def physical_record(config: dict) -> dict:
    """Calibrate on H0, then record independent H0/H1 rates and exact intervals."""
    calibration = calibrate_energy_threshold(
        capture(config, "noise", config["calibration_seed"], config["calibration_frames"]),
        config["target_pfa"], config["confidence_level"])
    result = {"calibration": asdict(calibration)}
    for label, kind in [("h0", "noise"), ("h1", "qpsk")]:
        energy = compute_energy_statistic(capture(config, kind, config[f"{label}_test_seed"], config[f"{label}_test_frames"]))
        count = int(np.count_nonzero(energy >= calibration.threshold))
        alternative = "less" if label == "h0" else "greater"
        result[label] = {"positive_decisions": count, "frames": len(energy), "rate": count / len(energy),
                         "exact_two_sided_interval": binomial_rate_interval(count, len(energy), config["confidence_level"]),
                         "exact_one_sided_interval": binomial_rate_interval(count, len(energy), config["confidence_level"], alternative)}
    result["acceptance_met"] = (result["h0"]["exact_one_sided_interval"][1] <= config["target_pfa"]
                                 and result["h1"]["exact_one_sided_interval"][0] >= config["target_pd"])
    return result


def network_record(config: dict) -> dict:
    """Record the known PU pulse intersections and unfinished-slot behavior."""
    results = {}
    for kind in ["csma", "pure", "slotted"]:
        engine = SimulationEngine(config["horizon_s"])
        if kind == "csma":
            mac = ILaCsmaMAC(engine, ChannelModel(random_seed=config["random_seed"]), max_retransmissions=0)
        elif kind == "pure":
            mac = PureAlohaMAC(engine, max_retransmissions=0)
        else:
            mac = SlottedAlohaMAC(engine, slot_duration_s=config["slot_duration_s"], max_retransmissions=0)
        pu = PrimaryUser(node_id=0, position=(1, 0), is_active=False)
        su = SecondaryUser(node_id=1, position=(0, 0))
        packet = Packet(packet_id=1, source_id=1)
        offset = config["slot_duration_s"] if kind == "slotted" else 0
        submit = mac.send_packet if kind == "pure" else mac.queue_packet
        submit(su, packet, [pu])
        engine.schedule_at(offset + config["pulse_on_s"], lambda: setattr(pu, "is_active", True))
        engine.schedule_at(offset + config["pulse_off_s"], lambda: setattr(pu, "is_active", False))
        engine.run()
        results[kind] = {"attempts": mac.packets_transmitted, "successful": mac.packets_successful,
                         "collided": mac.packets_collided_count, "dropped": mac.packets_dropped_count,
                         "pu_collisions": mac.pu_collisions,
                         "pu_overlap_s": mac.transmission_tracker.completed[0].pu_overlap_s}
    engine = SimulationEngine(config["short_horizon_s"])
    mac = SlottedAlohaMAC(engine, slot_duration_s=config["slot_duration_s"])
    packet = Packet(packet_id=1, source_id=1)
    mac.queue_packet(SecondaryUser(node_id=1, position=(0, 0)), packet)
    engine.run()
    results["unfinished_slot"] = {"attempts": mac.packets_transmitted, "successful": mac.packets_successful,
                                  "pending": len(mac.transmission_tracker.active), "receive_time": packet.receive_time}
    return results


def main() -> None:
    """Load an explicit YAML configuration and write an exclusive JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing run")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    bound = config["sample_bound"]
    rho = 10 ** (bound["uncertainty_db"] / 10)
    checks = []
    for snr in bound["snrs_db"]:
        count = int(compute_sample_complexity(snr, bound["uncertainty_db"], bound["target_pfa"], bound["target_pd"]))
        threshold = compute_theoretical_threshold(bound["target_pfa"], count, rho)
        # Exact worst-case v=1/rho, fixed pre-uncertainty signal power S.
        pd = float(stats.ncx2.sf(threshold * 2 * count * rho, 2 * count, 2 * count * 10 ** (snr / 10) * rho))
        checks.append({"snr_db": snr, "sufficient_samples": count, "exact_worst_case_pd": pd})
    result = {"scope": config["scope"], "limitations": ["No physical SDR or real-time validation",
              "No trained model or RadioML evaluation", "Network uses any-positive-overlap collision rule, not SINR"],
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for directory in ["math_models", "sdr_interface", "ns3_simulation"]
                                for p in sorted((ROOT / "src" / directory).glob("*.py"))},
              "sample_bound_checks": checks, "physical": physical_record(config["physical"]),
              "network": network_record(config["network"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["sample_bound_checks", "physical", "network"]}, indent=2))


if __name__ == "__main__":
    main()
