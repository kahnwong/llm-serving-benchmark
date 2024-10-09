#!/bin/bash

hyperfine --warmup 3 --runs 10 --export-json results/rs-mistralrs.json 'hurl tests/hurl/rs-mistralrs.hurl'
