#!/usr/bin/env python3

import os
import json
import flask
import importlib.util
import logging
import traceback
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import optimize

logger = logging.getLogger()
logger.setLevel(logging.WARN)


class RobustEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)


def not_found_on_error(handler):
    def new_handler(*args, **kwargs):
        try:
            res, status = handler(*args, **kwargs)
        except:
            e_repr = traceback.format_exc()
            logger.error(e_repr)
            res = {
                'state': 'UNAVAILABLE',
                'status': {'error_code': 'UNKNOWN', 'error_message': e_repr},
            }
            status = 404
        return flask.Response(
            response=json.dumps(res, cls=RobustEncoder),
            status=status,
            mimetype='application/json'
        )
    new_handler.__name__ = handler.__name__
    return new_handler

app = flask.Flask(__name__)


@app.route('/optimizer', methods=['GET'])
@not_found_on_error
def ping():
    status = 200
    res = {
        'optimizer_status': {
            'state': 'AVAILABLE',
            'status': {'error_code': 'OK', 'error_message': ''},
        }
    }
    return res, status


@app.route(
    '/optimizer/metadata',
    methods=['GET']
)
@not_found_on_error
def metadata():
    metadata = "test metadata"
    return metadata, 200


@app.route(
    '/optimizer:predict',
    methods=['POST']
)
@not_found_on_error
def predict():
    #body = flask.request.json

    guess0 = optimize.create_guess()
    guess1 = optimize.optimize1(guess0)
    guess2 = optimize.optimize2(guess1.x)

    output = {
        'guess0': str(guess0),
        'guess1': str(guess1.x),
        'guess2': str(guess2.x)
        }

    return output, 200


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return flask.Response(
        response='Model server is running!',
        status=200,
        mimetype='text/html'
    )
