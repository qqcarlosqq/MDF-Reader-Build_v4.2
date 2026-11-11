#!/usr/bin/env python3
import argparse
import platform
from pathlib import Path

APP_VERSION = "4.2.0"
DEFAULTS = {
    "rpm": "vFNRPM",
    "setpoint": "vNSET0",
    "alternator": "vALTLOAD",
    "learning_kblrc": "vKBLRC_[0]",
    "learning_isclrcb": "vISCLRCB",
    "coolant": "vTWN",
    "inst_af": "ES630_LA1_A_F",
    "ecm_af": "vAFSAF_[0]",
    "transition": "vSSTMD",
}


def clean(value):
    return str(value or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")


def read_text(path):
    return Path(path).read_text(encoding="utf-8-sig").strip()


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write("\t".join(clean(value) for value in row) + "\n")


def unique(values):
    return list(dict.fromkeys(values))


def read_overrides(path):
    result = {}
    if not path or not Path(path).exists():
        return result
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            result[(fields[0], fields[1])] = fields[2:]
    return result


def signal(mdf, label):
    locations = unique(mdf.whereis(label))
    if not locations:
        return None
    if len(locations) > 1:
        places = ", ".join(f"g{group}/c{channel}" for group, channel in locations[:8])
        raise ValueError(f"Duplicate label {label}: {places}")
    group, channel = locations[0]
    return mdf.get(group=group, index=channel, raw=False)


def arrays(sig):
    import numpy as np
    values = np.asarray(sig.samples)
    times = np.asarray(sig.timestamps, dtype=float)
    if values.dtype.kind not in "biuf":
        raise ValueError("Non-numeric signal")
    values = values.astype(float, copy=False)
    valid = np.isfinite(values) & np.isfinite(times)
    values = values[valid]
    times = times[valid]
    if values.size == 0:
        raise ValueError("No finite samples")
    order = np.argsort(times, kind="stable")
    return times[order], values[order]


def interpolate(times, values, target):
    import numpy as np
    if target < times[0] or target > times[-1]:
        return None
    return float(np.interp(target, times, values))


def window(times, values, start, end):
    import numpy as np
    if start is None or end is None or end < start:
        return np.array([]), np.array([])
    grid = np.unique(np.concatenate(([start, end], times[(times >= start) & (times <= end)])))
    return grid, np.interp(grid, times, values)


def first_up(times, values, threshold, after=None):
    import numpy as np
    above = values >= threshold
    indices = np.flatnonzero(above & np.r_[True, ~above[:-1]])
    for index in indices:
        if after is not None and times[index] < after:
            continue
        if index > 0 and values[index] != values[index - 1]:
            return float(times[index - 1] + (threshold - values[index - 1]) / (values[index] - values[index - 1]) * (times[index] - times[index - 1]))
        return float(times[index])
    return None


def first_down(times, values, threshold, after=None):
    import numpy as np
    below = values <= threshold
    indices = np.flatnonzero(below & np.r_[True, ~below[:-1]])
    for index in indices:
        if after is not None and times[index] < after:
            continue
        if index > 0 and values[index] != values[index - 1]:
            return float(times[index - 1] + (threshold - values[index - 1]) / (values[index] - values[index - 1]) * (times[index] - times[index - 1]))
        return float(times[index])
    return None


def resolved_label(mdf, recording, role, override_map, missing):
    configured = override_map.get((recording, role), [])
    candidates = []
    if configured:
        candidates.append(configured[0])
    candidates.append(DEFAULTS[role])
    if role == "rpm":
        candidates.append("vKNRPM")
    for candidate in unique([item for item in candidates if item]):
        if signal(mdf, candidate) is not None:
            return candidate
    missing.append(["MISSING", recording, role, DEFAULTS[role]])
    return None


def vSSTMD_events(times, values):
    """FIR is value 11. Exit is first change away from 11. Entry is first change in a run that eventually reaches 11."""
    events = []
    count = len(values)
    index = 1
    while index < count:
        previous = values[index - 1]
        current = values[index]
        if previous == 11 and current != 11:
            events.append((float(times[index]), "FIR -> NO FIR"))
        elif previous != 11 and current != previous:
            look = index
            reached_fir = False
            while look < count:
                if values[look] == 11:
                    reached_fir = True
                    break
                if look > index and values[look] == values[look - 1] and times[look] - times[index] > 2.0:
                    break
                look += 1
            if reached_fir:
                events.append((float(times[index]), "NO FIR -> FIR"))
                index = look
        index += 1
    return events


def exact_transition_events(times, values, fir_value, nonfir_value, entry_offset):
    events = []
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        if previous == fir_value and current == nonfir_value:
            events.append((float(times[index]), "FIR -> NO FIR"))
        elif previous == nonfir_value and current == fir_value:
            events.append((float(times[index]) + entry_offset, "NO FIR -> FIR"))
    return events


def analyze_recording(MDF, file_path, config, override_map):
    import numpy as np
    path = Path(file_path)
    recording = path.name
    missing = []
    base = [""] * 23
    base[0] = recording
    transitions = []
    try:
        with MDF(str(path)) as mdf:
            labels = {role: resolved_label(mdf, recording, role, override_map, missing) for role in DEFAULTS}
            if missing:
                return None, [], missing

            rpm_label = labels["rpm"]
            setpoint_label = labels["setpoint"]
            coolant_label = labels["coolant"]
            rpm_t, rpm_v = arrays(signal(mdf, rpm_label))
            set_t, set_v = arrays(signal(mdf, setpoint_label))
            coolant_t, coolant_v = arrays(signal(mdf, coolant_label))

            initial_time = float(rpm_t[0])
            base[15] = f'{labels["learning_kblrc"]}; {labels["learning_isclrcb"]}'
            base[16] = coolant_label
            base[17] = rpm_label
            base[18] = format(float(rpm_v[0]), ".15g")
            base[19] = setpoint_label
            base[21] = labels["inst_af"]
            base[22] = labels["ecm_af"]

            for column, role in [(1, "learning_kblrc"), (2, "learning_isclrcb")]:
                learning_t, learning_v = arrays(signal(mdf, labels[role]))
                # Learning channels may start after the RPM master timestamp. Use the
                # first recorded physical sample rather than requiring exact t=0 overlap.
                base[column] = format(float(learning_v[0]), ".15g")

            start_time = first_up(rpm_t, rpm_v, config["start_rpm"])
            if start_time is None:
                raise ValueError("Engine start not found")

            coolant_start = interpolate(coolant_t, coolant_v, start_time)
            if coolant_start is not None:
                base[3] = format(coolant_start, ".15g")

            flare_t, flare_v = window(rpm_t, rpm_v, start_time, min(start_time + 8.0, float(rpm_t[-1])))
            flare_index = int(np.argmax(flare_v))
            flare_peak_time = float(flare_t[flare_index])
            base[4] = format(float(flare_v[flare_index]), ".15g")

            over_start = first_up(rpm_t, rpm_v, config["flare_threshold"], start_time)
            over_end = first_down(rpm_t, rpm_v, config["flare_threshold"], over_start + 1e-9 if over_start is not None else None)
            if over_start is not None:
                if over_end is None:
                    over_end = float(rpm_t[-1])
                base[5] = format(max(0.0, over_end - over_start), ".15g")

            common_grid = np.unique(np.concatenate((rpm_t, set_t)))
            common_grid = common_grid[(common_grid >= flare_peak_time) & (common_grid <= min(rpm_t[-1], set_t[-1]))]
            delta = np.interp(common_grid, rpm_t, rpm_v) - np.interp(common_grid, set_t, set_v)
            below = delta < 0
            below_indices = np.flatnonzero(below & np.r_[True, ~below[:-1]])
            drop_start = float(common_grid[below_indices[0]]) if below_indices.size else None
            drop_min_time = None
            if drop_start is not None:
                drop_mask = (common_grid >= drop_start) & (common_grid <= drop_start + 3.0)
                local_indices = np.flatnonzero(drop_mask)
                minimum_index = local_indices[int(np.argmin(delta[drop_mask]))]
                drop_min_time = float(common_grid[minimum_index])
                base[6] = format(float(-delta[minimum_index]), ".15g")

            alternator_t, alternator_v = arrays(signal(mdf, labels["alternator"]))
            if drop_start is not None and drop_min_time is not None:
                _, values = window(alternator_t, alternator_v, drop_start, drop_min_time)
                if values.size:
                    base[7] = format(float(np.max(values)), ".15g")
                    base[8] = format(float(np.min(values)), ".15g")

            battery = signal(mdf, "vVB")
            start_switch = signal(mdf, "fSTSW")
            if battery is not None and start_switch is not None:
                vb_t, vb_v = arrays(battery)
                sw_t, sw_v = arrays(start_switch)
                drop_index = int(np.argmin(np.diff(vb_v)))
                t0 = float(vb_t[drop_index + 1])
                t1 = first_down(sw_t, sw_v, 0.0, t0)
                base[20] = "vVB; fSTSW"
            else:
                positive_steps = np.flatnonzero(np.diff(rpm_v) > 0)
                t0 = float(rpm_t[int(positive_steps[0]) + 1]) if positive_steps.size else float(rpm_t[0])
                t1 = first_up(rpm_t, rpm_v, 1000.0, t0)
                base[20] = f"{rpm_label} first step; {rpm_label} >= 1000 rpm"
            if t1 is not None:
                base[9] = format(max(0.0, t1 - t0), ".15g")

            for minimum_column, maximum_column, role in [(10, 11, "inst_af"), (12, 13, "ecm_af")]:
                af_t, af_v = arrays(signal(mdf, labels[role]))
                _, af_window = window(af_t, af_v, start_time, min(start_time + 2.0, float(af_t[-1])))
                if af_window.size:
                    base[minimum_column] = format(float(np.min(af_window)), ".15g")
                    base[maximum_column] = format(float(np.max(af_window)), ".15g")

            transition_label = labels["transition"]
            transition_override = override_map.get((recording, "transition"), [])
            transition_t, transition_v = arrays(signal(mdf, transition_label))
            if transition_override and len(transition_override) >= 4:
                transition_label = transition_override[0]
                fir_value = float(transition_override[1])
                nonfir_value = float(transition_override[2])
                entry_offset = float(transition_override[3])
                transition_t, transition_v = arrays(signal(mdf, transition_label))
                events = exact_transition_events(transition_t, transition_v, fir_value, nonfir_value, entry_offset)
            elif transition_label == "vSSTMD":
                events = vSSTMD_events(transition_t, transition_v)
            else:
                events = exact_transition_events(transition_t, transition_v, 8.0, 13.0, -0.6)

            base[14] = transition_label
            startup_entry_skipped = False
            for event_time, direction in events:
                # Ignore the first NO FIR -> FIR transition occurring during startup.
                # Later transitions of either direction remain included.
                if (not startup_entry_skipped and direction == "NO FIR -> FIR" and
                        event_time <= start_time + 3.0):
                    startup_entry_skipped = True
                    continue
                if event_time < max(rpm_t[0], set_t[0]) or event_time > min(rpm_t[-1], set_t[-1]):
                    continue
                initial_delta = interpolate(rpm_t, rpm_v, event_time) - interpolate(set_t, set_v, event_time)
                end_time = min(event_time + 6.0, float(rpm_t[-1]), float(set_t[-1]))
                grid = np.unique(np.concatenate(([event_time, end_time], rpm_t[(rpm_t >= event_time) & (rpm_t <= end_time)], set_t[(set_t >= event_time) & (set_t <= end_time)])))
                future_delta = np.interp(grid, rpm_t, rpm_v) - np.interp(grid, set_t, set_v)
                overspeed_gap = float(np.max(future_delta) - initial_delta)
                underspeed_gap = float(initial_delta - np.min(future_delta))
                coolant_value = interpolate(coolant_t, coolant_v, event_time)
                transitions.append([
                    format(initial_delta, ".15g"),
                    format(overspeed_gap, ".15g"),
                    format(underspeed_gap, ".15g"),
                    format(coolant_value, ".15g") if coolant_value is not None else "",
                ])

            return base, transitions, []
    except Exception as exc:
        base[14] = "ERROR: " + clean(exc)
        return base, [], []


def precheck_recordings(MDF, files, override_map):
    """Return grouped missing roles before any calculation starts."""
    grouped = {}
    for file_path in files:
        recording = Path(file_path).name
        try:
            with MDF(str(file_path)) as mdf:
                for role, default_label in DEFAULTS.items():
                    configured = override_map.get((recording, role), [])
                    candidates = []
                    if configured:
                        candidates.append(configured[0])
                    candidates.append(default_label)
                    if role == "rpm":
                        candidates.append("vKNRPM")
                    if not any(signal(mdf, candidate) is not None for candidate in unique([x for x in candidates if x])):
                        key = (role, default_label)
                        grouped.setdefault(key, []).append(recording)
        except Exception as exc:
            grouped.setdefault(("file_error", clean(exc)), []).append(recording)
    rows = []
    for (role, default_label), recordings in grouped.items():
        rows.append(["MISSING_GROUP", role, default_label, "|".join(recordings)])
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    for name in ["file-list", "config-file", "overrides-file", "output", "missing-output"]:
        parser.add_argument("--" + name)
    args = parser.parse_args()

    if args.self_test:
        import asammdf
        import numpy
        print("MDF Reader v" + APP_VERSION)
        print(platform.architecture()[0])
        print("SELF_TEST_OK")
        return 0

    from asammdf import MDF
    config = {}
    for line in read_text(args.config_file).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            config[key] = float(value)
    override_map = read_overrides(args.overrides_file)
    files = [line.strip() for line in read_text(args.file_list).splitlines() if line.strip()]
    if args.check_only:
        grouped_missing = precheck_recordings(MDF, files, override_map)
        write_tsv(args.missing_output, grouped_missing)
        return 4 if grouped_missing else 0
    records = []
    missing = []
    max_transitions = 0
    for file_path in files:
        base, transitions, problems = analyze_recording(MDF, file_path, config, override_map)
        missing.extend(problems)
        if base is not None:
            records.append((base, transitions))
            max_transitions = max(max_transitions, len(transitions))

    fixed_headers = [
        "Recording Name", "Learning vKBLRC_[0]", "Learning vISCLRCB", "Starting coolant temperature",
        "Engine flare", f"Flare duration above {config['flare_threshold']:g} rpm", "Drop",
        "MAX Alternator load", "MIN Alternator load", "Time to start", "MIN instrumented A/F",
        "MAX instrumented A/F", "MIN ECM A/F", "MAX ECM A/F", "Transition signal",
        "Learning signals", "Coolant temperature signal", "Engine speed signal", "Initial engine speed",
        "ISC setpoint", "Signals used for time to start", "Instrumented A/F", "ECM A/F"
    ]
    transition_headers = []
    for index in range(1, max_transitions + 1):
        transition_headers.extend([
            f"Transition {index} speed gap at change [rpm]",
            f"Transition {index} overspeed increase next 6 s [rpm]",
            f"Transition {index} underspeed increase next 6 s [rpm]",
            f"Transition {index} coolant temperature"
        ])
    rows = [fixed_headers + transition_headers]
    for base, transitions in records:
        flat = []
        for transition in transitions:
            flat.extend(transition)
        flat.extend([""] * (4 * max_transitions - len(flat)))
        rows.append(base + flat)
    write_tsv(args.output, rows)
    write_tsv(args.missing_output, missing)
    return 4 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
