#!/bin/bash

hyperfine --warmup 3 --runs 10 --export-json results/cpp-llamacpp.json 'hurl tests/hurl/cpp-llamacpp.hurl'
