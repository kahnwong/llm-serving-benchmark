#!/bin/bash

hyperfine --warmup 3 --runs 10 --export-json results/py-huggingface.json 'hurl tests/hurl/py-huggingface.hurl'
