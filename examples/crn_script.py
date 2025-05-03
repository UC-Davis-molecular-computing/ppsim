import polars as pl
import numpy as np
from matplotlib import pyplot as plt
import ppsim as pp

def main():
    a, b, u = pp.species('A B U')
    approx_majority = [
        a + b >> 2 * u,
        a + u >> 2 * a,
        b + u >> 2 * b,
    ]
    # init = {a: 50_001_000, b: 49_990_000}
    n = 10**9
    a_init = int(n * 0.51)
    b_init = n - a_init
    init = {a: a_init, b: b_init}
    # init = {a: 6, b: 4}
    sim = pp.Simulation(init, approx_majority, seed=1)
    # i, = species('I')
    # epidemic = [ i+u >> 2*i ]
    # init = { i:1, u:9 }
    # sim = Simulation(init, epidemic, seed=0)
    sim.run(10)
    # sim.history.plot()
    # plt.title('approximate majority protocol')
    # plt.xlim(0, sim.times[-1])
    # plt.ylim(0, sum(init.values()))
    # plt.savefig('examples/approx_majority_plot.png')
    # print("Plot saved to examples/approx_majority_plot.png")
    print(f"history =\n{sim.history}")


    num_multibatch_steps = sum(sim.simulator.collision_counts.values())
    num_collisions = 0
    digits_steps = 1
    digits_collisions = 1
    for collision_count, steps in sim.simulator.collision_counts.items():
        num_collisions += steps * collision_count
        digits_collisions = max(digits_collisions, len(str(num_collisions)))
        digits_steps = max(digits_steps, len(str(steps)))
    print(f'collision counts = (total: {num_collisions}, num_multibatch_steps: {num_multibatch_steps})')  
    for steps in sorted(sim.simulator.collision_counts.keys()):
        print(f'{sim.simulator.collision_counts[steps]:{digits_steps}} multibatch steps with {steps} collisions')
    
    # for count in sorted(sim.simulator.collision_counts.keys()):
    #     print(f'  {count}: {count*sim.simulator.collision_counts[count]:{digits_collisions}} collisions')
    # print('-'*20)
    
    
    sim.simulator.write_profile()
    

def main3():
    # derived rate constants of the formal reaction simulated by DNA strand displacement (units of /M/s)
    k1,k2,k3 = 9028, 2945, 1815
    total_concentration = 80 * 1e-9 # 1x volume was 80 nM
    vol = 50e-6 # 50 uL
    n = pp.concentration_to_count(total_concentration, vol)
    a,b,u = pp.species('A B U')
    approx_majority_rates = [
        (a+b >> 2*u).k(k1, units=pp.RateConstantUnits.mass_action),
        (a+u >> 2*a).k(k2, units=pp.RateConstantUnits.mass_action),
        (b+u >> 2*b).k(k3, units=pp.RateConstantUnits.mass_action),
    ]
    # set the initial concentrations near where the the mass-action CRN would reach an unstable equilibrium
    p = 0.45
    inits = {a: int(p*n), b: int((1-p)*n)}
    print(f'{inits=}')
    sim = pp.Simulation(inits, approx_majority_rates, volume=vol, time_units='seconds', seed=0)
    # print('delta:')
    # m = ['a', 'b', 'u']
    # for row in sim.simulator.delta:
    #     for i,j in row:
    #         print(f'({m[i]},{m[j]})', end=', ')
    #     print()
    # print('random_transitions:')
    # for row in sim.simulator.random_transitions: # type: ignore
    #     print(row)
    # print('random_outputs:')
    # for idx, (o1,o2) in enumerate(sim.simulator.random_outputs): # type: ignore
    #     print(f'idx {idx}: ({m[o1]},{m[o2]})', end=', ')
    # print()
    # print(f'{sim.simulator.transition_probabilities=}') # type: ignore
    sim.run()
    print(f"history =\n{sim.history}")
    sim.simulator.write_profile()


def dsd_oscillator():
    from ppsim import species, Simulation, RateConstantUnits, concentration_to_count
    # Fig. 1 in https://www.biorxiv.org/content/10.1101/138420v2.full.pdf
    # A+B --> 2B
    # B+C --> 2C
    # C+A --> 2A

    # signal species (represent formal species in formal CRN above)
    # index indicates whether it was the first or second product of a previous reaction
    b1, b2, c1, c2, a1, a2 = pp.species('b1  b2  c1  c2  a1  a2')

    signal_species = [b1, b2, c1, c2, a1, a2]

    # fuel species react step
    react_a_b_b1, back_a_b = species('react_a_b_b1  back_a_b')
    react_b_c_c1, back_b_c = species('react_b_c_c1  back_b_c')
    react_c_a_a1, back_c_a = species('react_c_a_a1  back_c_a')

    react_species = [react_a_b_b1, react_b_c_c1, react_c_a_a1]
    back_species = [back_a_b, back_b_c, back_c_a]

    # fuel species produce step
    produce_b_b1_b2, helper_b_b2 = species('produce_b_b1_b2  helper_b_b2')
    produce_c_c1_c2, helper_c_c2 = species('produce_c_c1_c2  helper_c_c2')
    produce_a_a1_a2, helper_a_a2 = species('produce_a_a1_a2  helper_a_a2')

    produce_species = [produce_b_b1_b2, produce_c_c1_c2, produce_a_a1_a2]
    helper_species = [helper_b_b2, helper_c_c2, helper_a_a2]
    fuel_species = react_species + produce_species

    # intermediate species
    flux_b_b1, flux_c_c1, flux_a_a1 = species('flux_b_b1  flux_c_c1  flux_a_a1')
    reactint_a1_b_b1, reactint_b1_c_c1, reactint_c1_a_a1 = species('reactint_a1_b_b1  reactint_b1_c_c1  reactint_c1_a_a1') 
    reactint_a2_b_b1, reactint_b2_c_c1, reactint_c2_a_a1 = species('reactint_a2_b_b1  reactint_b2_c_c1  reactint_c2_a_a1') 
    productint_b_b1_b2, productint_c_c1_c2, productint_a_a1_a2 = species('productint_b_b1_b2  productint_c_c1_c2  productint_a_a1_a2')

    flux_species = [flux_b_b1, flux_c_c1, flux_a_a1]
    reactint_species = [reactint_a1_b_b1, reactint_b1_c_c1, reactint_c1_a_a1,
                        reactint_a2_b_b1, reactint_b2_c_c1, reactint_c2_a_a1]
    produceint_species = [productint_b_b1_b2, productint_c_c1_c2, productint_a_a1_a2]

    # waste species react step
    waste_a1_b1, waste_a1_b2, waste_a2_b1, waste_a2_b2 = species('waste_a1_b1  waste_a1_b2  waste_a2_b1  waste_a2_b2')
    waste_b1_c1, waste_b1_c2, waste_b2_c1, waste_b2_c2 = species('waste_b1_c1  waste_b1_c2  waste_b2_c1  waste_b2_c2')
    waste_c1_a1, waste_c1_a2, waste_c2_a1, waste_c2_a2 = species('waste_c1_a1  waste_c1_a2  waste_c2_a1  waste_c2_a2')

    # waste species produce step
    waste_b_b1_b2, waste_c_c1_c2, waste_a_a1_a2 = species('waste_b_b1_b2  waste_c_c1_c2  waste_a_a1_a2')

    waste_species = [waste_a1_b1, waste_a1_b2, waste_a2_b1, waste_a2_b2,
                    waste_b1_c1, waste_b1_c2, waste_b2_c1, waste_b2_c2,
                    waste_c1_a1, waste_c1_a2, waste_c2_a1, waste_c2_a2,
                    waste_b_b1_b2, waste_c_c1_c2, waste_a_a1_a2]

    # DSD reactions implementing formal CRN
    # A+B --> 2B
    ab_react_rxns = [
        a1 + react_a_b_b1 | back_a_b + reactint_a1_b_b1,
        a2 + react_a_b_b1 | back_a_b + reactint_a2_b_b1,
        reactint_a1_b_b1 + b1 >> waste_a1_b1 + flux_b_b1, # typo in Fig. 1; these rxns irreversible
        reactint_a1_b_b1 + b2 >> waste_a1_b2 + flux_b_b1, #
        reactint_a2_b_b1 + b1 >> waste_a2_b1 + flux_b_b1, #
        reactint_a2_b_b1 + b2 >> waste_a2_b2 + flux_b_b1, #
    ]
    ab_produce_rxns = [
        flux_b_b1 + produce_b_b1_b2 | b1 + productint_b_b1_b2,
        helper_b_b2 + productint_b_b1_b2 >> waste_b_b1_b2 + b2,
    ]
    ab_rxns = ab_react_rxns + ab_produce_rxns

    # B+C --> 2C
    bc_react_rxns = [
        b1 + react_b_c_c1 | back_b_c + reactint_b1_c_c1,
        b2 + react_b_c_c1 | back_b_c + reactint_b2_c_c1,
        reactint_b1_c_c1 + c1 >> waste_b1_c1 + flux_c_c1,
        reactint_b1_c_c1 + c2 >> waste_b1_c2 + flux_c_c1,
        reactint_b2_c_c1 + c1 >> waste_b2_c1 + flux_c_c1,
        reactint_b2_c_c1 + c2 >> waste_b2_c2 + flux_c_c1,
    ]
    bc_produce_rxns = [
        flux_c_c1 + produce_c_c1_c2 | c1 + productint_c_c1_c2,
        helper_c_c2 + productint_c_c1_c2 >> waste_c_c1_c2 + c2,
    ]
    bc_rxns = bc_react_rxns + bc_produce_rxns

    # C+A --> 2A
    ca_react_rxns = [
        c1 + react_c_a_a1 | back_c_a + reactint_c1_a_a1,
        c2 + react_c_a_a1 | back_c_a + reactint_c2_a_a1,
        reactint_c1_a_a1 + a1 >> waste_c1_a1 + flux_a_a1,
        reactint_c1_a_a1 + a2 >> waste_c1_a2 + flux_a_a1,
        reactint_c2_a_a1 + a1 >> waste_c2_a1 + flux_a_a1,
        reactint_c2_a_a1 + a2 >> waste_c2_a2 + flux_a_a1,
    ]
    ca_produce_rxns = [
        flux_a_a1 + produce_a_a1_a2 | a1 + productint_a_a1_a2,
        helper_a_a2 + productint_a_a1_a2 >> waste_a_a1_a2 + a2,
    ]
    ca_rxns = ca_react_rxns + ca_produce_rxns

    all_rps_dsd_rxns = ab_rxns + bc_rxns + ca_rxns

    all_species = signal_species + \
                react_species + \
                back_species + \
                produce_species + \
                helper_species + \
                flux_species + \
                reactint_species + \
                produceint_species + \
                waste_species

    # These functions map states to categories, which allow HistoryPlotter to show a simplified plot of categories
    def aux(state):
        if state in react_species:
            return 'react'
        if state in produce_species:
            return 'produce'
        if state in waste_species:
            return 'waste'
        if state in helper_species:
            return 'helper'
        
    def abc(state):
        if state in signal_species:
            return state.name[0]
    
    
    uL = 10 ** -6  # 1 uL (microliter)
    nL = 10 ** -9
    nM = 10 ** -9  # 1 nM (nanomolar)

    k = 1e6  # forward rate constant in mass-action units
    r = 1e6  # reverse rate constant in mass-action units
    for rxn in all_rps_dsd_rxns:
        rxn.k(k, units=RateConstantUnits.mass_action)
        if rxn.reversible:
            rxn.r(r, units=RateConstantUnits.mass_action)

    vol = 10 * nL

    # scale time to make simulations take less time
    time_scaling = 1
    vol /= time_scaling

    react_conc = 100 * nM
    back_conc = 100 * nM
    helper_conc = 75 * nM
    produce_conc = 100 * nM
    a1_conc = 11 * nM
    b1_conc = 10 * nM
    c1_conc = 3 * nM

    # this factor scales all concentrations
    conc_factor = 1

    react_count = concentration_to_count(react_conc * conc_factor, vol)
    back_count = concentration_to_count(back_conc * conc_factor , vol)
    helper_count = concentration_to_count(helper_conc* conc_factor, vol)
    produce_count = concentration_to_count(produce_conc* conc_factor, vol)
    a1_count = concentration_to_count(a1_conc* conc_factor, vol)
    b1_count = concentration_to_count(b1_conc* conc_factor, vol)
    c1_count = concentration_to_count(c1_conc* conc_factor, vol)

    init_config_react = {specie: react_count for specie in react_species}
    init_config_back = {specie: back_count for specie in back_species}
    init_config_helper = {specie: helper_count for specie in helper_species}
    init_config_produce = {specie: produce_count for specie in produce_species}

    init_config = {a1: a1_count, b1: b1_count, c1: c1_count}
    init_config.update(init_config_react)
    init_config.update(init_config_back)
    init_config.update(init_config_helper)
    init_config.update(init_config_produce)

    sim = Simulation(init_config=init_config, rule=all_rps_dsd_rxns, volume=vol, time_units='seconds')
    print(f'{init_config=}')
    hours = 12
    sim.run(hours * 3600, 60)  # run for 12 hours, saving every 60 seconds
    print(sim.history)
    sim.simulator.write_profile()


def main2():
    a,b,u = pp.species('A B U')
    approx_majority = [
        a+b >> 2*u,
        a+u >> 2*a,
        b+u >> 2*b,
    ]
    n = 10 ** 2
    p = 0.51
    a_init = int(n * p)
    b_init = n - a_init
    init = {a: a_init, b: b_init}
    # for seed in range(100):
    #     print(f'{seed=}')
    seed = 10
    sim = pp.Simulation(init, approx_majority, seed=seed, 
                        # simulator_method='sequential'
                        )
    # sim.run(20, 1)
    # sim.run(100)
    sim.run(5)
    print(sim.history)

def sample_configs():
    a,b,u = pp.species('A B U')
    approx_majority = [
        a+b >> 2*u,
        a+u >> 2*a,
        b+u >> 2*b,
    ]

    trials_exponent = 6
    pop_exponent = 4
    n = 10 ** pop_exponent
    p = 0.51 # TODO: restore this
    # p = 0.5
    a_init = int(n * p)
    b_init = n - a_init
    inits = {a: a_init, b: b_init}
    trials = 10 ** trials_exponent
    end_time = 5
    sim = pp.Simulation(inits, approx_majority)

    fn_noext = f'examples/rebop_samples_popsize10e{pop_exponent}_trials10e{trials_exponent}'
    fn = f'{fn_noext}.parquet'
    results_rebop = pl.read_parquet(fn)
    results_ppsim = sim.sample_future_configuration(end_time, num_samples = trials)

    fig, ax = plt.subplots(figsize = (10,4))
    state = 'A'
    # state = 'B'
    # state = 'U'
    ax.hist([results_ppsim[state], results_rebop[state]], 
            bins = np.linspace(int(n*0.32), int(n*.43), 20), # type: ignore
            alpha = 1, label=['ppsim', 'rebop']) #, density=True, edgecolor = 'k', linewidth = 0.5)
    ax.legend()

    ax.set_xlabel(f'count of state {state}')
    ax.set_ylabel(f'empirical probability')
    ax.set_title(f'state {state} distribution sampled at simulated time {end_time} ($10^{trials_exponent}$ samples)')
    
    # plt.ylim(0, 200_000)

    pdf_fn = f'{fn_noext}_regula.pdf'
    plt.savefig(pdf_fn, bbox_inches='tight')
    plt.show()
    

if __name__ == '__main__':
    # main2()
    sample_configs()
