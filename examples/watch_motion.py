"""Consume numeric motion states from the local service, using only stdlib."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/api/v1/events")
    parser.add_argument("--frames", type=int, default=0, help="0 streams until interrupted")
    args = parser.parse_args()
    count = 0
    try:
        with urllib.request.urlopen(args.url, timeout=30) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                state = json.loads(line[5:])
                print(json.dumps({
                    "sequence": state["sequence"],
                    "tracking_level": state.get("tracking_level"),
                    "face_center": state.get("face_center"),
                    "face_motion": state.get("face_motion"),
                    "left_arm": state.get("left_arm"),
                    "right_arm": state.get("right_arm"),
                    "events": state.get("events"),
                }), flush=True)
                count += 1
                if args.frames and count >= args.frames:
                    break
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
