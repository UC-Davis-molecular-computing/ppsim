from ppsim.crn import species, stochkit_format, write_stochkit_file

def main():
    a, b, u = species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        (a + u | 2 * a).k(2).r(3),
        b + u >> 2 * b,
    ]
    n = 100
    init_config = {a: round(0.6*n), b: round(0.4*n)}
    s = stochkit_format(approx_majority, init_config, n, 'Approximate Approximate Majority')
    print(s)
    print('writing to file test_stochkit.xml')
    write_stochkit_file('test_stochkit.xml', approx_majority, init_config, n, 'Approximate Approximate Majority')

if __name__ == '__main__':
    main()