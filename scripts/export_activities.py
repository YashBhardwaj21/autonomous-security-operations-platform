from pathlib import Path
import json

from src.canon.schema import SourceType
from src.ingestion.otrf import iter_scenarios, read_raw_events
from src.ingestion.parser import ParserFactory, DropStats
from src.sessions.session_builder import SessionBuilder

OUTPUT_DIR = Path("data/processed/activities")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def entity_to_dict(entity):
    """Convert dataclass/Pydantic/object into a JSON-serializable dict."""

    if entity is None:
        return None

    if hasattr(entity, "model_dump"):
        return entity.model_dump()

    if hasattr(entity, "__dict__"):
        return vars(entity)

    return entity


def activity_to_dict(activity):
    """Convert an Activity object into a JSON dict."""
    if hasattr(activity, "model_dump"):
        return activity.model_dump(mode="json")

    return {
        "activity_id": getattr(activity, "activity_id", None),
        "scenario_id": getattr(activity, "scenario_id", None),
        "host": getattr(activity, "host", None),
        "logon_id": getattr(activity, "logon_id", None),
        "source": getattr(activity, "source", "OTRF"),
        "start_time": getattr(activity, "start_time", None),
        "end_time": getattr(activity, "end_time", None),
        "processes": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "processes", {}).items()
        },
        "users": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "users", {}).items()
        },
        "files": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "files", {}).items()
        },
        "registry": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "registry", {}).items()
        },
        "network": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "network", {}).items()
        },
        "services": {
            str(k): entity_to_dict(v)
            for k, v in getattr(activity, "services", {}).items()
        },
        "relationships": getattr(activity, "relationships", []),
    }


def main():

    factory = ParserFactory()
    stats = DropStats()
    builder = SessionBuilder(factory)

    exported = 0

    for scenario_id, meta, zips in iter_scenarios():

        print(f"\nProcessing Scenario: {scenario_id}")

        events = []

        for zp in zips:

            for raw in read_raw_events(zp):

                ev = factory.parse(raw, SourceType.OTRF, stats)

                if ev is not None:
                    events.append(ev)

        print(f"Loaded {len(events)} parsed events")

        if not events:
            continue

        activities = list(
            builder.build_sessions(
                events,
                scenario_id=scenario_id,
            )
        )

        print(f"Built {len(activities)} activities")


        for activity in activities:

            outfile = OUTPUT_DIR / f"activity_{exported:06d}.json"

            with outfile.open("w", encoding="utf-8") as f:

                json.dump(
                    activity_to_dict(activity),
                    f,
                    indent=2,
                    default=str,
                )

            exported += 1

    print("\n====================================")
    print(f"Exported {exported} activities")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")
    print("====================================")


if __name__ == "__main__":
    main()