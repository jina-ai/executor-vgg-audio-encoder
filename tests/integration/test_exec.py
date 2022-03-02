__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pytest
from pathlib import Path

from jina import Document, DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='function')
def config_file():
    return str(Path(__file__).parents[0] / 'flow.yml')


def test_embedding_exists(sample_file, config_file):
    fn = f'{sample_file}.wav'
    doc = DocumentArray([Document(uri=fn)])
    with Flow.load_config(config_file) as f:
        resp = f.post(on='/index', inputs=doc, return_results=True)
    assert resp[0].embedding is not None
