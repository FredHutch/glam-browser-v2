#!/usr/bin/env python3
"""Command line utility to modify the GLAM Browser database."""
import argparse
from .db import GLAM_DB
from .index import GLAM_INDEX
import logging
import os

def open_glam_db(args):
    return GLAM_DB(
        db_name=args.db_name,
        db_username=args.db_username,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
    )

def glam_index(fp=None, uri=None):
    GLAM_INDEX(
        input_hdf=fp,
        output_base=uri,
    ).index()

def main():
    """Entrypoint for the CLI."""

    # Set up logging
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [GLAM CLI] %(message)s'
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    # Write to STDOUT
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(prog='GLAM Browser Command Line Utility')

    # Arguments needed for all subcommands
    parser.add_argument(
        "--db-name", 
        type=str, 
        help="Name of database", 
        default=os.environ.get("DB_NAME", "glam_db")
    )
    parser.add_argument(
        "--db-username", 
        type=str, 
        help="Username for database",
        default=os.environ.get("DB_USERNAME")
    )
    parser.add_argument(
        "--db-password", 
        type=str, 
        help="Password for database",
        default=os.environ.get("DB_PASSWORD")
    )
    parser.add_argument(
        "--db-host", 
        type=str, 
        help="Host for database",
        default=os.environ.get("DB_HOST")
    )
    parser.add_argument(
        "--db-port", 
        type=str, 
        help="Port for database",
        default=os.environ.get("DB_PORT")
    )

    subparsers = parser.add_subparsers(help='Select a command')

    # SET UP THE DATABASE
    parser_add_user = subparsers.add_parser("setup", help="Set up a database")
    parser_add_user.set_defaults(
        func=lambda args: open_glam_db(args).setup()
    )

    # ADD A NEW USER
    parser_add_user = subparsers.add_parser("add-user", help="Add a user")
    parser_add_user.add_argument("--username", type=str, help="Username for new user")
    parser_add_user.add_argument("--email", type=str, help="Email for new user")
    parser_add_user.set_defaults(
        func=lambda args: open_glam_db(args).add_user(username=args.username, email=args.email)
    )

    # SET USER PASSWORD
    parser_set_password = subparsers.add_parser("set-password", help="Set the password for a user")
    parser_set_password.add_argument("--username", type=str, help="Username to set password for")
    parser_set_password.add_argument("--password", type=str, help="New password")
    parser_set_password.set_defaults(
        func=lambda args: open_glam_db(args).set_password(username=args.username, password=args.password)
    )

    # INDEX A DATASET
    parser_index_dataset = subparsers.add_parser("index-dataset", help="Index a dataset")
    parser_index_dataset.add_argument("--fp", type=str, help="Path to input dataset in HDF5 format")
    parser_index_dataset.add_argument("--uri", type=str, help="Path to output dataset in AWS S3")
    parser_index_dataset.set_defaults(
        func=lambda args: glam_index(fp=args.fp, uri=args.uri)
    )


    # ADD A DATASET
    parser_add_dataset = subparsers.add_parser("add-dataset", help="Add a dataset")
    parser_add_dataset.add_argument("--dataset-id", type=str, help="Unique ID for dataset")
    parser_add_dataset.add_argument("--name", type=str, help="Human readable name for dataset")
    parser_add_dataset.add_argument("--uri", type=str, help="Path to dataset")
    parser_add_dataset.set_defaults(
        func=lambda args: open_glam_db(args).add_dataset(dataset_id=args.dataset_id, name=args.name, uri=args.uri)
    )

    # GRANT ACCESS TO A DATASET
    parser_grant_access = subparsers.add_parser("grant-access", help="Allow a user to access a dataset")
    parser_grant_access.add_argument("--user-name", type=str, help="Name of user")
    parser_grant_access.add_argument("--dataset-name", type=str, help="Name of dataset")
    parser_grant_access.set_defaults(
        func=lambda args: open_glam_db(args).grant_access(user=args.user_name, dataset=args.dataset_name)
    )

    # DUMP TABLE CONTENTS
    parser_grant_access = subparsers.add_parser("dump-table", help="Print the contents of any table")
    parser_grant_access.add_argument("--table", type=str, help="Name of the table to read")
    parser_grant_access.set_defaults(
        func=lambda args: print(open_glam_db(args).read_table(args.table))
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the command which is linked to the selected subcommand
    args.func(args)


if __name__ == "__main__":

    # Run the main entrypoint
    main()
