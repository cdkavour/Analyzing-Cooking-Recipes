import matplotlib.pyplot as plt
import json

def main():
    times = json.load(open("times.json"))
    Times = []
    for i in times:
        if int(times[i] <= 1440):
            Times.append(int(times[i]))
    Times = sorted(Times)
    buck = 0
    buckets = []
    while buck <= 1440:
        buckets.append(buck)
        buck += 10

    hist = plt.hist(x=Times, bins=buckets)
    plt.xlabel("Ready In Time in Minutes")
    plt.ylabel("Count")
    plt.title("Distribution of Ready In Times")
    plt.show()

    return

if __name__ == '__main__':
    main()
