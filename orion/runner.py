"""Orion runner.

Functions responsible for executing Pipelines on Signals
and storing the results inside an Orion Database using
the Orion Explorer.
"""
import logging
import pickle

LOGGER = logging.getLogger(__name__)


def get_outputs_spec(pipeline):
    outputs = ["default"]
    try:
        output_names = pipeline.get_output_names('visualization')
        outputs.append('visualization')
    except ValueError:
        output_names = []

    return outputs, output_names


def process_pipeline_output(orex, signalrun, pipeline_output, output_names):
    if not isinstance(pipeline_output, tuple):
        return pipeline_output

    events = pipeline_output[0]
    if output_names:
        # There might be multiple `default` outputs before the `visualization`
        # outputs in the pipeline_output tuple, thus we get the last entries
        # corresponding to visualization
        visualization = pipeline_output[-len(output_names):]
        visualization_dict = dict(zip(output_names, visualization))
        for name, value in visualization_dict.items():
            kwargs = {
                "filename": '{}-{}.pkl'.format(signalrun.id, name),
                "signalrun_id": signalrun.id,
                "variable": name
            }
            with orex._fs.new_file(**kwargs) as f:
                pickle.dump(value, f)

    return events


def start_signalrun(orex, datarun, signal):
    signalrun = orex.add_signalrun(datarun, signal)
    signalrun.start()
    LOGGER.info('Signalrun %s started', signalrun.id)

    try:
        data = signalrun.signal.load()
        pipeline = signalrun.datarun.pipeline
        mlpipeline = pipeline.load()
        outputs, output_names = get_outputs_spec(mlpipeline)

        LOGGER.info('Running pipeline %s on signal %s', pipeline.name, signal.name)
        pipeline_output = mlpipeline.fit(data, output_=outputs)

        # LOGGER.info('Producing pipeline %s on signal %s', pipeline.name, signal.name)
        # pipeline_output = mlpipeline.predict(data, output_=outputs)

        LOGGER.info('Processing pipeline %s predictions on signal %s', pipeline.name, signal.name)
        events = process_pipeline_output(orex, signalrun, pipeline_output, output_names)
        status = signalrun.STATUS_SUCCESS

    except Exception:
        LOGGER.exception('Signalrun %s crashed', signalrun.id)
        events = []
        status = signalrun.STATUS_ERRORED

    signalrun.end(status, events)


def start_datarun(orex, experiment, pipeline):
    """Start executing a Datarun and store the results on DB.

    Args:
        orex (OrionExplorer):
            OrionExplorer instance to use to store the results
            inside the Database.
        experiment (Experiment or ObjectId or str):
            The Experiment to which the created Datarun will
            belong.
        pipeline (Pipeline or ObjectId or str):
            Pipeline to use for the Datarun.
    """
    datarun = orex.add_datarun(experiment, pipeline)
    datarun.start()
    LOGGER.info('Datarun %s started', datarun.id)

    try:
        for signal in experiment.signals:
            start_signalrun(orex, datarun, signal)

        status = datarun.STATUS_SUCCESS

    except Exception:
        LOGGER.exception('Datarun %s crashed', datarun.id)
        status = datarun.STATUS_ERRORED

    datarun.end(status)
