# -*- coding: utf-8 -*-

import getpass
import json
import logging
from datetime import datetime

from pymongo import MongoClient

from orion.utils import remove_dots, restore_dots

LOGGER = logging.getLogger(__name__)


def MongoDB(object):

    def __init__(self, database=None, config=None, **kwargs):
        if config:
            with open(config, 'r') as f:
                config = json.load(f)
        else:
            config = kwargs

        host = config.get('host', 'localhost')
        port = config.get('port', 27017)
        user = config.get('user')
        password = config.get('password')
        database = database or config.get('database', 'test')
        auth_database = config.get('auth_database', 'admin')

        if user and not password:
            password = getpass.getpass(prompt='Please insert database password: ')

        client = MongoClient(
            host=host,
            port=port,
            username=user,
            password=password,
            authSource=auth_database
        )

        LOGGER.info("Setting up a MongoClient %s", client)

        self._db = client[database]

    def load_template(self, template_name):
        match = {
            'name': template_name
        }

        cursor = self._db.templates.find(match)
        templates = list(cursor.sort('insert_ts', -1).limit(1))

        if templates:
            return restore_dots(templates[0])

    def insert_template(self, template):
        if 'name' not in template:
            raise ValueError("Templates need to have a name key")

        template['insert_ts'] = datetime.utcnow()
        template = remove_dots(template)

        self._db.templates.insert_one(template)

    def insert_pipeline(self, candidate, score, dataset, table, column):

        pipeline = candidate.to_dict()

        pipeline['score'] = score
        pipeline['dataset'] = dataset
        pipeline['table'] = table
        pipeline['column'] = column
        pipeline['insert_ts'] = datetime.utcnow()

        pipeline = remove_dots(pipeline)

        self._db.pipelines.insert_one(pipeline)
