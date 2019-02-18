#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os

import pandas as pd
from mlblocks import MLPipeline

from orion.data import load_signal
from orion.utils import logging_setup

LOGGER = logging.getLogger(__name__)


def process_signal(signal, output, args):
    LOGGER.info("Loading signal %s", signal)
    data = load_signal(signal, args.timestamp_column, args.value_column)
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


def get_signals(input_path):
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            yield os.path.join(input_path, filename)


def get_parser():
    parser = argparse.ArgumentParser(description='Orion Command Line Interface.')
    parser.add_argument('-l', '--logfile',
                        help='Name of the logfile. If not given, log to stdout.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')

    parser.add_argument('-o', '--output', default='output',
                        help='Path to the folder were anomaly files will be written')

    parser.add_argument('-T', '--timestamp-column', type=int,
                        help='Index of the column to be used as timestamp')
    parser.add_argument('-V', '--value-column', type=int,
                        help='Index of the column to be used as the value')

    parser.add_argument('-p', '--pipeline', required=True,
                        help='Name or path of the pipeline to use')

    parser.add_argument('signals', nargs='*',
                        help='Name or path of the signals to process')
    parser.add_argument('-i', '--input',
                        help='Path to the folder were the signal files will be read from')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.signals and not args.input:
        parser.error('Please provide at least a signal name or an --input path')

    logging_setup(args.verbose, args.logfile)

    os.makedirs(args.output, exist_ok=True)
    process_signals(args)
