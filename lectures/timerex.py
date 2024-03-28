from timer import timer # type: ignore

print("Elapsed time for computing the sum of first",
    "ten million positive integers is")
stopwatch = timer()
stopwatch.start()
sum = 0
for k in range(1, 100000001):
    sum += k
stopwatch.stop()
print("{:.15e}".format(stopwatch.get_elapsed_time), " seconds.\n")