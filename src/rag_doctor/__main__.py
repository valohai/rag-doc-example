import argparse
import logging
import os
import sys

import valohai
from dotenv import load_dotenv

from rag_doctor.consts import DEFAULT_DATABASE_DIR
from rag_doctor.database import create_database, get_qdrant_client
from rag_doctor.interactive import start_chat

VECTORIZE_CMD = "create-database"
CHAT_CMD = "chat"


def cli(sys_argv: list[str]) -> int:
    program_name = os.path.basename(sys_argv[0])
    usage_msg = f"Usage: {program_name} [{VECTORIZE_CMD}|{CHAT_CMD}]"

    if len(sys_argv) < 2:
        print(usage_msg)
        return 1

    status = 0
    command = sys_argv[1]
    if command == VECTORIZE_CMD:
        cli_create_database(sys_argv)
    elif command == CHAT_CMD:
        cli_chat(sys_argv)
    else:
        print(usage_msg)
        if command != "--help":
            print(f"Unknown command: {command}")
            status = 1

    return status


def cli_create_database(sys_argv: list[str]) -> None:
    # fmt: off
    parser = argparse.ArgumentParser()
    doc_dir_on_valohai = valohai.inputs("documentation_csv").dir_path()
    parser.add_argument("--documentation_dir", type=str, default=doc_dir_on_valohai, help="Path to directory containing documentation CSV files")
    parser.add_argument("--content_column_index", type=int, default=0, help="Index of the document content column in the CSV files")
    parser.add_argument("--source_column_index", type=int, default=1, help="Index of the source link column in the CSV files")
    parser.add_argument("--header_row_skip", type=int, default=0, help="Number of initial rows to skip in the CSV files i.e. the header rows")
    parser.add_argument("--database_dir", type=str, default=DEFAULT_DATABASE_DIR, help="Path to directory to store Qdrant vector database")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    create_database(
        documentation_dir=args.documentation_dir,
        content_column_index=args.content_column_index,
        source_column_index=args.source_column_index,
        header_row_skip=args.header_row_skip,
        database_dir=args.database_dir,
    )


def cli_chat(sys_argv: list[str]) -> None:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_dir", type=str, default=DEFAULT_DATABASE_DIR, help="Path to directory containing Qdrant vector database")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    db_client = get_qdrant_client(args.database_dir)
    start_chat(db_client=db_client)


def main():
    load_dotenv()  # load environment variables from `.env`
    logging.basicConfig(level=logging.INFO)
    try:
        status = cli(sys.argv)
    except KeyboardInterrupt:
        print("\nExiting...")
        status = 0
    return status


if __name__ == "__main__":
    main()
