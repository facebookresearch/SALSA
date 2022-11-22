# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

from .lattice import LatticeEnvironment


logger = getLogger()


ENVS = {
    'lattice': LatticeEnvironment,
}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)

    # tasks
    tasks = [x for x in params.tasks.split(',') if len(x) > 0]
    assert len(tasks) == len(set(tasks)) > 0
    assert all(task in env.TRAINING_TASKS for task in tasks)
    params.tasks = tasks
    logger.info(f'Training tasks: {", ".join(tasks)}')

    return env
