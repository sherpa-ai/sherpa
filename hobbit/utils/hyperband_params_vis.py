import math


def visualize_hyperband_params():
    R = input("Enter R: ") or 81
    eta = input("Enter eta (default 3): ") or 3
    R = float(R)
    eta = float(eta)

    log_eta = lambda x: math.log(x) / math.log(eta)
    s_max = int(log_eta(R))
    B = (s_max + 1) * R

    run_line = []
    header_line = ['stage']
    stages = [['{}'.format(i)] for i in range(s_max+1)]

    for s in reversed(range(s_max + 1)):
        n = int(math.ceil(B / R / (s + 1) * eta ** s))
        r = R * eta ** (-s)

        run_line.append('run={}'.format(s_max - s + 1))
        header_line.append('models\tepochs')

        for i in range(s + 1):
            n_i = int(n * eta ** (-i))
            r_i = int(round(r * eta ** (i)))

            stages[i].append('{}\t{}'.format(n_i, r_i))

    print('\t' + '\t\t'.join(run_line))
    print('\t'.join(header_line))
    for stage in stages:
        print('\t'.join(stage))


if __name__ == '__main__':
    visualize_hyperband_params()