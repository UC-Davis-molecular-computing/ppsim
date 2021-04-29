from ppsim.crn import species, stochkit_format, write_stochkit_file

def main():
    a, b, u = species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]
    init_config = {a: 60, b: 40}
    s = stochkit_format(approx_majority, init_config, 'Approximate Majority')
    print(s)
    print('writing to file test_stochkit.xml')
    write_stochkit_file('test_stochkit.xml', approx_majority, init_config, 'Approximate Majority')

if __name__ == '__main__':
    main()