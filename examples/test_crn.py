
from ppsim import species, Simulation
from matplotlib import pyplot as plt

def main():
    a, b, u = species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]
    n = 10 ** 5
    init_config = {a: 0.51 * n, b: 0.49 * n}
    sim = Simulation(init_config, approx_majority)
    sim.run()
    sim.history.plot()
    plt.title('4 state majority protocol')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlim(0, sim.times[-1])
    plt.ylim(0, n)

if __name__ == '__main__':
    main()