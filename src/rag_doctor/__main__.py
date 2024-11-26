import argparse
import logging
import os
import sys

import valohai
from dotenv import load_dotenv

from rag_doctor.database import create_database, prepare_database
from rag_doctor.interactive import start_chat
from rag_doctor.query import create_rag_chain

CREATE_DB_CMD = "create-database"
QUERY_CMD = "query"
CHAT_CMD = "chat"
SERVE_CMD = "serve"


def cli(sys_argv: list[str]) -> int:
    program_name = os.path.basename(sys_argv[0])
    usage_msg = f"Usage: {program_name} [{CREATE_DB_CMD}|{QUERY_CMD}|{CHAT_CMD}|{SERVE_CMD}]"

    if len(sys_argv) < 2:
        print(usage_msg)
        return 1

    status = 0
    command = sys_argv[1]
    if command == CREATE_DB_CMD:
        cli_create_database(sys_argv)
    elif command == CHAT_CMD:
        cli_chat(sys_argv)
    elif command == QUERY_CMD:
        cli_query(sys_argv)
    elif command == SERVE_CMD:
        cli_serve(sys_argv)
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
    parser.add_argument("--database_dir", type=str, default="./qdrant_data", help="Path to directory to store Qdrant vector database")
    parser.add_argument("--documentation_dir", type=str, default=doc_dir_on_valohai, help="Path to directory containing documentation CSV files")
    parser.add_argument("--content_column_index", type=int, default=0, help="Index of the document content column in the CSV files")
    parser.add_argument("--source_column_index", type=int, default=1, help="Index of the source link column in the CSV files")
    parser.add_argument("--header_row_skip", type=int, default=0, help="Number of initial rows to skip in the CSV files i.e. the header rows")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    create_database(
        documentation_dir=args.documentation_dir,
        content_column_index=args.content_column_index,
        source_column_index=args.source_column_index,
        header_row_skip=args.header_row_skip,
        database_dir=args.database_dir,
    )


def cli_query(sys_argv: list[str]) -> None:
    # fmt: off
    db_dir_on_valohai = valohai.inputs("embedding_db").dir_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_dir", type=str, default=db_dir_on_valohai, help="Path to directory containing Qdrant vector database")
    parser.add_argument("--question", type=str, required=True, help="Question to ask", action="append")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    questions = args.question

    db_client = prepare_database(args.database_dir)
    rag_chain = create_rag_chain(db_client)

    for question in questions:
        print("\nQuestion: ")
        print(question)

        message = rag_chain(question)
        if not message or not message.content:
            raise ValueError("No response from the model")

        print("\nAnswer: ")
        print(message.content)


def cli_chat(sys_argv: list[str]) -> None:
    # fmt: off
    db_dir_on_valohai = valohai.inputs("embedding_db").dir_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_dir", type=str, default=db_dir_on_valohai, help="Path to directory containing Qdrant vector database")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    db_client = prepare_database(args.database_dir)
    start_chat(db_client=db_client)


def cli_serve(sys_argv: list[str]) -> None:
    # fmt: off
    db_dir_on_valohai = valohai.inputs("embedding_db").dir_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_dir", type=str, default=db_dir_on_valohai, help="Path to directory containing Qdrant vector database")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    args, _ = parser.parse_known_args(sys_argv[2:])
    # fmt: on

    import uvicorn
    from rag_doctor.server import create_app

    db_client = prepare_database(args.database_dir)
    app = create_app(db_client)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    load_dotenv()  # load environment variables from `.env`
    debug_value = os.getenv("DEBUG", "0").lower()
    logging_level = logging.DEBUG if debug_value not in ("0", "false") else logging.WARNING
    logging.basicConfig(level=logging_level)
    try:
        status = cli(sys.argv)
    except KeyboardInterrupt:
        print("\nExiting...")
        status = 0
    return status


if __name__ == "__main__":
    main()
