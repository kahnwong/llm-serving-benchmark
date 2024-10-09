import json


def parse_stats(filename: str):
    # read file
    with open(f"../results/{filename}", "r") as f:
        data = json.load(f)

    # set framework name
    framework = filename.split(".")[0]

    # extract times
    times = data["results"][0]["times"]

    # extract stats
    stats = data["results"][0]
    stats.pop("times")
    stats.pop("command")
    stats.pop("stddev")
    stats.pop("user")
    stats.pop("system")
    stats.pop("exit_codes")

    return framework, times, stats


def extract_stats():
    # define result files
    filenames = [
        "cpp-llamafile.json",
        "cpp-llamacpp.json",
        "go-ollama.json",
        "py-huggingface.json",
        "rs-mistralrs.json",
    ]

    # init variables
    stats_all = []
    times_all = []

    for filename in filenames:
        framework, times, stats = parse_stats(filename)

        # stats
        stats.update({"framework": framework})
        stats_all.append(stats)

        # time
        for index, time in enumerate(times):
            times_all.append(
                {
                    "framework": framework,
                    "index": index,
                    "time": time,
                }
            )

    return times_all, stats_all
