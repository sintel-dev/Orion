#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os

import pandas as pd
from mlblocks import MLPipeline

from orion.analysis import analyze
from orion.data import load_signal
from orion.explorer import OrionExplorer
from orion.utils import logging_setup

LOGGER = logging.getLogger(__name__)


def process_signal(signal, output, args):
    LOGGER.info("Loading signal %s", signal)
    data = load_signal(signal, None, args.timestamp_column, args.value_column)
    LOGGER.info("Signal shape: %s", data.shape)

    LOGGER.info("Loading pipeline %s", args.pipeline)
    pipeline = MLPipeline.load(args.pipeline)

    LOGGER.info("Fitting the pipeline")
    pipeline.fit(data)

    LOGGER.info("Finding anomalies")
    anomalies = pipeline.predict(data)

    LOGGER.info("%s Anomalies found", len(anomalies))

    adf = pd.DataFrame(anomalies, columns=['start', 'end', 'score'])
    adf['start'] = adf['start'].astype(int)
    adf['end'] = adf['end'].astype(int)

    LOGGER.info("Storing Anomalies as %s", output)
    adf.to_csv(output, index=False)


def process_signals(args):
    for signal in args.signals:
        if signal.endswith('.csv'):
            signal_name = os.path.basename(signal)[:-4]
        else:
            signal_name = signal

        output = os.path.join(args.output, signal_name + '.anomalies.csv')

        try:
            LOGGER.info("Processing signal %s", signal)
            process_signal(signal, output, args)
            LOGGER.info("Signal %s processed succcessfully", signal)
        except Exception:
            LOGGER.exception("Exception processing signal %s", signal)


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

    explorer.add_dataset(
        args.name,
        args.signal,
        args.satellite,
        args.location,
        args.timestamp_column,
        args.value_column
    )


def _add_pipeline(explorer, args):
    explorer.add_pipeline(
        args.name,
        args.path,
    )


def _add_comment(explorer, args):
    explorer.add_comment(
        args.event,
        args.text,
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
    else:
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
    datarun = analyze(explorer, args.dataset, args.pipeline)
    print('Datarun id: {}'.format(datarun.id))


def get_parser():

    # Common Parent - Shared options
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('-D', '--database', default='orion',
                        help='Name of the database to connect to. Defaults to "orion"')
    common.add_argument('-L', '--limit', type=int, help='Limit the number of results')
    common.add_argument('-l', '--logfile',
                        help='Name of the logfile. If not given, log to stdout.')
    common.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')

    parser = argparse.ArgumentParser(description='Orion Command Line Interface.')
    parser.set_defaults(function=None)

    action = parser.add_subparsers(title='action', dest='action')
    action.required = True

    reset = action.add_parser('reset', help='Reset (drop) the database', parents=[common])
    reset.set_defaults(action=_reset)

    # add
    add = action.add_parser('add', help='Add an object to the database', parents=[common])
    add_model = add.add_subparsers(title='model', dest='model')
    add_model.required = True

    # Add dataset
    add_dataset = add_model.add_parser('dataset', help='Add a new dataset', parents=[common])
    add_dataset.set_defaults(function=_add_dataset)

    add_dataset.add_argument('-T', '--timestamp-column', default=0, type=int,
                             help='Position of the timestamp column in the CSV,')
    add_dataset.add_argument('-V', '--value-column', default=0, type=int,
                             help='Position of the value column in the CSV,')
    add_dataset.add_argument('-s', '--signal',
                             help='Name or ID of the signal. Defaults to the given `name`')
    add_dataset.add_argument('-S', '--satellite',
                             help='Name or ID of the satellite. Defaults to `None`')
    add_dataset.add_argument('name', help='Name of this dataset')
    add_dataset.add_argument('location', nargs='?', help='path to the CSV file')

    # Add pipeline
    add_pipeline = add_model.add_parser('pipeline', help='Add a new pipeline', parents=[common])
    add_pipeline.set_defaults(function=_add_pipeline)
    add_pipeline.add_argument('name', help='Name of this pipeline')
    add_pipeline.add_argument('path', help='path to the JSON file')

    # Add comment
    add_comment = add_model.add_parser('comment', parents=[common],
                                       help='Add a comment to an event')
    add_comment.set_defaults(function=_add_comment)
    add_comment.add_argument('event', help='ID of the event')
    add_comment.add_argument('text', help='Comment text')

    # list
    list_parent = argparse.ArgumentParser(add_help=False)
    list_parent.add_argument('-o', '--output',
                             help='Dump the output into the given CSV path.')

    list_ = action.add_parser('list', parents=[common, list_parent],
                              help='List objects from the database')
    list_model = list_.add_subparsers(title='model', dest='model')
    list_model.required = True
    list_.set_defaults(function=_list)
    list_.set_defaults(delete=[])
    list_.set_defaults(filters=[])

    # list datasets
    list_datasets = list_model.add_parser('datasets', parents=[common, list_parent],
                                          help='List datasets')
    list_datasets.set_defaults(model='datasets')

    # list pipelines
    list_pipelines = list_model.add_parser('pipelines', parents=[common, list_parent],
                                           help='List pipelines')
    list_pipelines.set_defaults(model='pipelines')
    list_pipelines.set_defaults(delete=['mlpipeline'])

    # list dataruns
    list_dataruns = list_model.add_parser('dataruns', parents=[common, list_parent],
                                          help='List dataruns')
    list_dataruns.add_argument('-d', '--dataset', help='ID of the dataset')
    list_dataruns.add_argument('-p', '--pipeline', help='ID of the pipeline')
    list_dataruns.set_defaults(model='dataruns')
    list_dataruns.set_defaults(filters=['dataset', 'pipeline'])

    # list events
    list_events = list_model.add_parser('events', parents=[common, list_parent],
                                        help='List found events')
    list_events.add_argument('-d', '--datarun', nargs='?', help='ID of the datarun')
    list_events.set_defaults(model='events')
    list_events.set_defaults(filters=['datarun'])

    # list comments
    list_comments = list_model.add_parser('comments', parents=[common, list_parent],
                                          help='List event comments')
    list_comments.add_argument('-d', '--datarun', nargs='?', help='ID of the datarun')
    list_comments.add_argument('-e', '--event', nargs='?', help='ID of the event')
    list_comments.set_defaults(model='comments')
    list_comments.set_defaults(filters=['datarun', 'event'])

    # Analyze
    run = action.add_parser('run', help='Run a pipeline on a dataset', parents=[common])
    run.add_argument('pipeline', help='ID or name of the pipeline')
    run.add_argument('dataset', help='ID of name of the dataset')
    run.set_defaults(function=_run)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging_setup(args.verbose, args.logfile)

    explorer = OrionExplorer(args.database)

    args.function(explorer, args)
