total_times = {}

def add_time_and_print(time, name):
    global total_times
    n_calls = 1
    if(name in total_times):
        time += total_times[name][0]
        n_calls += total_times[name][1]
    total_times[name] = (time, n_calls)

    print(f"{name}: total time = {time}, calls = {n_calls}")