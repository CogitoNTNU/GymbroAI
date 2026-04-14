import argparse
import csv
import re
from pathlib import Path


def _find_exercise_files(data_dir: Path, exercise_name: str):
    pattern = re.compile(rf"^{re.escape(exercise_name)}(\d+)\.csv$", re.IGNORECASE)
    matched = []

    for csv_file in data_dir.glob("*.csv"):
        match = pattern.match(csv_file.name)
        if match:
            matched.append((int(match.group(1)), csv_file))

    matched.sort(key=lambda item: item[0])
    return [path for _, path in matched]


def _detect_exercise_prefixes(data_dir: Path):
    pattern = re.compile(r"^([a-zA-Z_]+)\d+\.csv$", re.IGNORECASE)
    prefixes = set()
    for csv_file in data_dir.glob("*.csv"):
        match = pattern.match(csv_file.name)
        if match:
            prefixes.add(match.group(1).lower())
    return sorted(prefixes)


def concatenate_exercise_csv(
    data_dir: Path,
    exercise_name: str,
    dry_run: bool = False,
    delete_sources: bool = True,
):
    files = _find_exercise_files(data_dir, exercise_name)
    if not files:
        raise FileNotFoundError(
            f"No numbered CSV files found for exercise '{exercise_name}' in {data_dir}"
        )

    output_file = data_dir / f"{exercise_name}.csv"
    temp_output_file = data_dir / f".{exercise_name}.tmp.csv"

    header = None
    total_rows = 0
    deleted_files = []

    if not dry_run:
        with temp_output_file.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)

            for input_file in files:
                with input_file.open("r", newline="", encoding="utf-8") as in_handle:
                    reader = csv.reader(in_handle)
                    input_header = next(reader, None)

                    if input_header is None:
                        continue

                    if header is None:
                        header = input_header
                        writer.writerow(header)
                    elif input_header != header:
                        raise ValueError(f"Header mismatch in file: {input_file.name}")

                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1

        temp_output_file.replace(output_file)

        if delete_sources:
            for input_file in files:
                input_file.unlink()
                deleted_files.append(input_file.name)
    else:
        for input_file in files:
            with input_file.open("r", newline="", encoding="utf-8") as in_handle:
                reader = csv.reader(in_handle)
                input_header = next(reader, None)

                if input_header is None:
                    continue

                if header is None:
                    header = input_header
                elif input_header != header:
                    raise ValueError(f"Header mismatch in file: {input_file.name}")

                for _ in reader:
                    total_rows += 1

    return {
        "exercise": exercise_name,
        "output": str(output_file),
        "files": [path.name for path in files],
        "rows": total_rows,
        "dry_run": dry_run,
        "deleted_files": deleted_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine numbered exercise CSV files (e.g. curl1.csv, curl2.csv) "
            "into a single file named <exercise>.csv."
        )
    )
    parser.add_argument(
        "exercise",
        nargs="?",
        help="Exercise name prefix, e.g. curl",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing the CSV files. Defaults to this script's folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and count rows without writing output.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Combine all detected exercise groups in one run.",
    )
    parser.add_argument(
        "--keep-sources",
        action="store_true",
        help="Keep numbered source files after combining.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    detected = _detect_exercise_prefixes(data_dir)

    process_all = args.all or not args.exercise

    if process_all:
        if not detected:
            raise FileNotFoundError(
                f"No numbered exercise CSV files found in {data_dir}"
            )

        for exercise_name in detected:
            summary = concatenate_exercise_csv(
                data_dir,
                exercise_name,
                dry_run=args.dry_run,
                delete_sources=not args.keep_sources,
            )
            print(f"Exercise: {summary['exercise']}")
            print(
                f"Matched files ({len(summary['files'])}): {', '.join(summary['files'])}"
            )
            print(f"Total data rows: {summary['rows']}")
            if summary["dry_run"]:
                print("Dry run enabled: no output file written.")
            else:
                print(f"Output written: {summary['output']}")
                if args.keep_sources:
                    print("Source files kept.")
                else:
                    print(
                        f"Deleted source files ({len(summary['deleted_files'])}): "
                        f"{', '.join(summary['deleted_files'])}"
                    )
            print()
        return

    summary = concatenate_exercise_csv(
        data_dir,
        args.exercise,
        dry_run=args.dry_run,
        delete_sources=not args.keep_sources,
    )

    print(f"Exercise: {summary['exercise']}")
    print(f"Matched files ({len(summary['files'])}): {', '.join(summary['files'])}")
    print(f"Total data rows: {summary['rows']}")
    if summary["dry_run"]:
        print("Dry run enabled: no output file written.")
    else:
        print(f"Output written: {summary['output']}")
        if args.keep_sources:
            print("Source files kept.")
        else:
            print(
                f"Deleted source files ({len(summary['deleted_files'])}): "
                f"{', '.join(summary['deleted_files'])}"
            )


if __name__ == "__main__":
    main()
