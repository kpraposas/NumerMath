from timer import timer

stopwatch = timer()
stopwatch.start()
sum = 0
for k in range(1, 100000001):
    sum += k
stopwatch.stop()

print("Elapsed time for computing the sum of first")
print("ten million positive integers is", "{:.15e}".format(
    stopwatch.get_elapsed_time), " seconds.\n")