#!/bin/bash

hyperfine --warmup 3 --runs 10 --export-json results/cpp-llamafile.json 'hurl tests/hurl/cpp-llamafile.hurl'
