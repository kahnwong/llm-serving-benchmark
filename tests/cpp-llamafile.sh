#!/bin/bash

hyperfine --warmup 15 --runs 15 --export-json results/cpp-llamafile.json 'hurl tests/hurl/cpp-llamafile.hurl'
