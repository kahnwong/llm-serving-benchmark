#!/bin/bash

hyperfine --warmup 3 --runs 10 --export-json results/go-ollama.json 'hurl tests/hurl/go-ollama.hurl'
