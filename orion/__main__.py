#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import warnings

import tabulate

from orion.analysis import get_available_templates
from orion.benchmark import benchmark
from orion.evaluation import CONTEXTUAL_METRICS as METRICS

warnings.filterwarnings("ignore")


def _evaluate(args):
    if args.all:
        pipelines = get_available_templates()
    else:
        pipelines = args.pipeline

    scores = benchmark(pipelines=pipelines, datasets=args.signal, metrics=args.metric,
                       rank=args.rank, test_split=args.holdout)

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

    parser = argparse.ArgumentParser(description='Orion Command Line Interface.')
    parser.set_defaults(function=None)

    action = parser.add_subparsers(title='action', dest='action')
    action.required = True

    # Evaluate
    evaluate = action.add_parser('evaluate', parents=[base],
                                 help='Evaluate one or more pipelines on NASA signals')
    evaluate.add_argument('-s', '--signal', action='append',
                          help='Signal to use. Use multiple times for more signals.')
    evaluate.add_argument('-m', '--metric', action='append', default=METRICS,
                          help='Metric to use. Use multiple times for more metrics.')
    evaluate.add_argument('-r', '--rank', default='f1', help='Rank scores based on this metric.')
    evaluate.add_argument('-o', '--output', help='Write the results in the specified CSV file.')
    evaluate.add_argument('--holdout', dest='holdout', action='store_true', default=None,
                          help='Holdout test data during training.')
    evaluate.add_argument('--no-holdout', dest='holdout', action='store_false', default=None,
                          help='Do not holdout test data curing training.')
    evaluate.set_defaults(function=_evaluate)

    group = evaluate.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pipeline', default=[], action='append',
                       help='Name of the pipeline JSONs to evaluate.')
    group.add_argument('-a', '--all', action='store_true', help='Evaluate all known pipelines.')

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
        args.function(args)


if __name__ == "__main__":
    main()
