#!/usr/bin/env python
# coding: utf-8

import warnings   # noqa isort:skip
warnings.filterwarnings("ignore")  # noqa isort:skip

import argparse
import getpass
import logging
import os
import sys
from urllib.error import HTTPError

import tabulate

from orion import PIPELINES
from orion.data import load_signal
from orion.db.explorer import OrionDBExplorer
from orion.evaluation import evaluate_pipelines


def _reset(explorer, args):
    print('WARNING: This will drop the database!')
    name = input('Please enter the database name to confirm: ')
    if name == args.database:
        print('Dropping database {}'.format(name))
        explorer.drop_database()
    else:
        print('Aborting.')


def _add_dataset(explorer, args):
    if args.signal is None:
        args.signal = args.name

    if not args.start or not args.stop:
        path_or_name = args.location or args.name

        try:
            data = load_signal(path_or_name, None, args.timestamp_column, args.value_column)
        except HTTPError:
            print('Unknown signal: {}'.format(path_or_name))
            sys.exit(1)
        else:
            timestamps = data['timestamp']
            if not args.start:
                args.start = timestamps.min()

            if not args.stop:
                args.stop = timestamps.max()

    explorer.add_dataset(
        args.name,
        args.signal,
        args.satellite,
        args.start,
        args.stop,
        args.location,
        args.timestamp_column,
        args.value_column,
        args.user,
    )


def _add_pipeline(explorer, args):
    try:
        explorer.add_pipeline(
            args.name,
            args.path,
            args.user,
        )
    except FileNotFoundError:
        print('File not found: {}'.format(args.path))
    except IsADirectoryError:
        print('File is a directory: {}'.format(args.path))


def _add_comment(explorer, args):
    explorer.add_comment(
        args.event,
        args.text,
        args.user,
    )


def _list(explorer, args):
    name = args.model
    kwargs = {
        name: getattr(args, name)
        for name in args.filters
    }

    method = getattr(explorer, 'get_' + name)
    documents = method(**kwargs)

    if documents.empty:
        print('No {} found'.format(name))
        return

    for delete_ in args.delete:
        del documents[delete_]

    if args.limit:
        documents = documents.head(args.limit)

    if args.output:
        print('Storing results in {}'.format(args.output))
        documents.to_csv(args.output, index=False)
    else:
        print(documents.to_string())


def _run(explorer, args):
    try:
        datarun = explorer.analyze(args.dataset, args.pipeline, args.user)
        print('Datarun id: {}'.format(datarun.id))
    except Exception as ex:
        print("There was an error processing the dataset {}: {}".format(args.dataset, ex))
        sys.exit(1)


def _process(explorer, args):
    pipeline_name = os.path.basename(args.pipeline)[:-5]
    print('Adding pipeline {}'.format(pipeline_name))
    explorer.add_pipeline(pipeline_name, args.pipeline, args.user)

    for path in args.paths:
        try:
            name = os.path.basename(path)
            if not name.endswith('.csv'):
                raise ValueError('Invalid CSV name: {}'.format('.csv'))

            name = name[:-4]

            print('Adding dataset {}'.format(name))
            explorer.add_dataset(
                name,
                name,
                location=path,
                timestamp_column=args.timestamp_column,
                value_column=args.value_column,
                user_id=args.user,
            )

            print('Processing CSV {}'.format(path))
            explorer.analyze(name, pipeline_name, args.user)

        except Exception as ex:
            print("Error processing CSV {}: {}".format(path, ex))


def _evaluate(args):
    if args.all:
        pipelines = PIPELINES
    else:
        pipelines = args.pipeline

    scores = evaluate_pipelines(pipelines, args.signal, args.metric, args.rank, args.holdout)

    if args.output:
        print('Writing results in {}'.format(args.output))
        scores.to_csv(args.output, index=False)

    print(tabulate.tabulate(
        scores,
        showindex=False,
        tablefmt='github',
        headers=scores.columns
    ))


def get_parser():

    # Common Parent - Shared options
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument('-l', '--logfile',
                      help='Name of the logfile. If not given, log to stdout.')
    base.add_argument('-v', '--verbose', action='count', default=0,
                      help='Be verbose. Use -vv for increased verbosity.')

    common = argparse.ArgumentParser(add_help=False, parents=[base])
    common.add_argument('-D', '--database', default='orion',
                        help='Name of the database to connect to. Defaults to "orion"')

    common_user = argparse.ArgumentParser(add_help=False, parents=[common])
    common_user.add_argument('-u', '--user', default=getpass.getuser(), help='User identifier')

    parser = argparse.ArgumentParser(description='Orion Command Line Interface.')
    parser.set_defaults(function=None)

    action = parser.add_subparsers(title='action', dest='action')
    action.required = True

    reset = action.add_parser('reset', help='Reset (drop) the database', parents=[common])
    reset.set_defaults(function=_reset)
    reset.set_defaults(user=None)

    # add
    add = action.add_parser('add', help='Add an object to the database')
    add_model = add.add_subparsers(title='model', dest='model')
    add_model.required = True

    # Add dataset
    add_dataset = add_model.add_parser('dataset', parents=[common_user], help='Add a new dataset')
    add_dataset.set_defaults(function=_add_dataset)

    add_dataset.add_argument('-T', '--timestamp-column', type=int,
                             help='Position of the timestamp column in the CSV,')
    add_dataset.add_argument('-V', '--value-column', type=int,
                             help='Position of the value column in the CSV,')
    add_dataset.add_argument('-s', '--signal',
                             help='Name or ID of the signal. Defaults to the given `name`')
    add_dataset.add_argument('-S', '--satellite',
                             help='Name or ID of the satellite. Defaults to `None`')
    add_dataset.add_argument('--start', type=int, help='Start time, as an integer unix timestamp')
    add_dataset.add_argument('--stop', type=int, help='Stop time, as an integer unix timestamp')
    add_dataset.add_argument('name', help='Name of this dataset')
    add_dataset.add_argument('location', nargs='?', help='path to the CSV file')

    # Add pipeline
    add_pipeline = add_model.add_parser('pipeline', parents=[common_user],
                                        help='Add a new pipeline')
    add_pipeline.set_defaults(function=_add_pipeline)
    add_pipeline.add_argument('name', help='Name of this pipeline')
    add_pipeline.add_argument('path', help='path to the JSON file')

    # Add comment
    add_comment = add_model.add_parser('comment', parents=[common_user],
                                       help='Add a comment to an event')
    add_comment.set_defaults(function=_add_comment)
    add_comment.add_argument('event', help='ID of the event')
    add_comment.add_argument('text', help='Comment text')

    # list
    common_list = argparse.ArgumentParser(add_help=False, parents=[common])
    common_list.add_argument('-o', '--output',
                             help='Dump the output into the given CSV path.')
    common_list.add_argument('-L', '--limit', type=int, help='Limit the number of results')

    list_ = action.add_parser('list', help='List objects from the database')
    list_model = list_.add_subparsers(title='model', dest='model')
    list_model.required = True
    list_.set_defaults(function=_list)
    list_.set_defaults(delete=[])
    list_.set_defaults(filters=[])
    list_.set_defaults(user=None)

    # list datasets
    list_datasets = list_model.add_parser('datasets', parents=[common_list],
                                          help='List datasets')
    list_datasets.set_defaults(model='datasets')

    # list pipelines
    list_pipelines = list_model.add_parser('pipelines', parents=[common_list],
                                           help='List pipelines')
    list_pipelines.set_defaults(model='pipelines')
    list_pipelines.set_defaults(delete=['mlpipeline'])

    # list dataruns
    list_dataruns = list_model.add_parser('dataruns', parents=[common_list],
                                          help='List dataruns')
    list_dataruns.add_argument('-d', '--dataset', help='ID of the dataset')
    list_dataruns.add_argument('-p', '--pipeline', help='ID of the pipeline')
    list_dataruns.set_defaults(model='dataruns')
    list_dataruns.set_defaults(filters=['dataset', 'pipeline'])

    # list events
    list_events = list_model.add_parser('events', parents=[common_list],
                                        help='List found events')
    list_events.add_argument('-d', '--datarun', nargs='?', help='ID of the datarun')
    list_events.set_defaults(model='events')
    list_events.set_defaults(filters=['datarun'])

    # list comments
    list_comments = list_model.add_parser('comments', parents=[common_list],
                                          help='List event comments')
    list_comments.add_argument('-d', '--datarun', nargs='?', help='ID of the datarun')
    list_comments.add_argument('-e', '--event', nargs='?', help='ID of the event')
    list_comments.set_defaults(model='comments')
    list_comments.set_defaults(filters=['datarun', 'event'])

    # Analyze
    run = action.add_parser('run', help='Run a pipeline on a dataset', parents=[common_user])
    run.add_argument('pipeline', help='ID or name of the pipeline')
    run.add_argument('dataset', help='ID of name of the dataset')
    run.set_defaults(function=_run)

    # Process
    process = action.add_parser('process', parents=[common_user],
                                help='Process one or more signal CSV files using a pipeline')
    process.add_argument('-T', '--timestamp-column', type=int,
                         help='Position of the timestamp column in the CSV,')
    process.add_argument('-V', '--value-column', type=int,
                         help='Position of the value column in the CSV,')
    process.add_argument('pipeline', help='Path to the pipeline JSON')
    process.add_argument('paths', nargs='+', help='Paths to the CSV files')
    process.set_defaults(function=_process)

    # Evaluate
    evaluate = action.add_parser('evaluate', parents=[base],
                                 help='Evaluate one or more pipelines on NASA signals')
    evaluate.add_argument('-s', '--signal', action='append',
                          help='Signal to use. Use multiple times for more signals.')
    evaluate.add_argument('-m', '--metric', action='append',
                          help='Metric to use. Use multiple times for more metrics.')
    evaluate.add_argument('-r', '--rank', help='Rank scores based on this metric.')
    evaluate.add_argument('-o', '--output', help='Write the results in the specified CSV file.')
    evaluate.add_argument('--holdout', dest='holdout', action='store_true', default=None,
                          help='Holdout test data during training.')
    evaluate.add_argument('--no-holdout', dest='holdout', action='store_false', default=None,
                          help='Do not holdout test data curing training.')
    group = evaluate.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--all', action='store_true', help='Evaluate all known pipelines.')
    group.add_argument('pipeline', default=[], nargs='*',
                       help='Paths to the pipeline JSONs to evaluate')

    return parser


def logging_setup(verbosity=1, logfile=None, logger_name=None):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging_setup(args.verbose, args.logfile)

    if args.action == 'evaluate':
        _evaluate(args)
    else:
        explorer = OrionDBExplorer(args.database)

        args.function(explorer, args)
