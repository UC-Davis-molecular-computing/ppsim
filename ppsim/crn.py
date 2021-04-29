"""
Module for expression population protocols using CRN notation. Ideas and much code taken from
https://github.com/enricozb/python-crn.
"""

from __future__ import annotations  # needed for forward references in type hints

from collections import defaultdict
import copy
from enum import Enum
from typing import Union, Dict, Tuple, Set, Iterable, DefaultDict, List
from dataclasses import dataclass
from xml.dom import minidom

# example of StochKit format:
'''
<Model>
   <Description>Epidemic</Description>
   <NumberOfReactions>1</NumberOfReactions>
   <NumberOfSpecies>2</NumberOfSpecies>
   <ParametersList>
     <Parameter>
       <Id>c1</Id>
       <Expression>1.0</Expression>
     </Parameter>
   </ParametersList>
   <ReactionsList>
     <Reaction>
       <Id>R2</Id>
       <Description> A+B -> 2B </Description>
       <Type>mass-action</Type>
       <Rate>c1</Rate>
       <Reactants>
           <SpeciesReference id="A" stoichiometry="1"/>
           <SpeciesReference id="B" stoichiometry="1"/>
       </Reactants>
       <Products>
           <SpeciesReference id="B" stoichiometry="2"/>
       </Products>
     </Reaction>
  </ReactionsList>
  <SpeciesList>
     <Species>
       <Id>A</Id>
       <Description>Species #1</Description>
       <InitialPopulation>10000</InitialPopulation>
     </Species>
     <Species>
       <Id>B</Id>
       <Description>Species #2</Description>
       <InitialPopulation>1</InitialPopulation>
     </Species>
  </SpeciesList>
</Model>
'''


def species_in_rxns(rxns: Iterable[Reaction]):
    """

    Args:
        rxns: iterable of :any:`Reaction`'s

    Returns: list of species (without repetitions) in :any:`Reaction`'s in `rxns`
    """
    species_set: Set[Specie] = set()
    species_list: List[Specie] = []
    for rxn in rxns:
        for sp in rxn.reactants.species + rxn.products.species:
            if sp not in species_set:
                species_set.add(sp)
                species_list.append(sp)
    return species_list


def stochkit_format(rxns: Iterable[Reaction], init_config: Dict[Specie, int], name: str = 'CRN') -> str:
    """

    Args:
        rxns: reactions to translate to StochKit format
        init_config: dict mapping each :any:`Specie` to its initial count
        name: name of the CRN

    Returns:
        string describing CRN in StochKit XML format
    """
    rxns = replace_reversible_rxns(rxns)
    species_list = species_in_rxns(rxns)

    root = minidom.Document()

    model = root.createElement('Model')
    root.appendChild(model)

    desc_node = root.createElement('Description')
    model.appendChild(desc_node)
    desc_text = root.createTextNode(name)
    desc_node.appendChild(desc_text)

    num_rxns_node = root.createElement('NumberOfReactions')
    model.appendChild(num_rxns_node)
    num_rxns_text = root.createTextNode(str(len(rxns)))
    num_rxns_node.appendChild(num_rxns_text)

    num_species_node = root.createElement('NumberOfSpecies')
    model.appendChild(num_species_node)
    num_species_text = root.createTextNode(str(len(species_list)))
    num_species_node.appendChild(num_species_text)

    species_node = root.createElement('SpeciesList')
    model.appendChild(species_node)
    for specie in species_list:
        specie_node = root.createElement('Species')
        species_node.appendChild(specie_node)

        id_node = root.createElement('Id')
        specie_node.appendChild(id_node)
        id_text = root.createTextNode(specie.name)
        id_node.appendChild(id_text)

        specie_description_node = root.createElement('Description')
        specie_node.appendChild(specie_description_node)
        specie_description_text = root.createTextNode(specie.name)
        specie_description_node.appendChild(specie_description_text)

        initial_population_node = root.createElement('InitialPopulation')
        specie_node.appendChild(initial_population_node)
        count = init_config.get(specie, 0)
        initial_population_text = root.createTextNode(str(count))
        initial_population_node.appendChild(initial_population_text)

    rxns_node = root.createElement('ReactionsList')
    model.appendChild(rxns_node)
    for idx, rxn in enumerate(rxns):
        rxn_node = root.createElement('Reaction')
        rxns_node.appendChild(rxn_node)

        id_node = root.createElement('Id')
        rxn_node.appendChild(id_node)
        id_text = root.createTextNode(f'R{idx}')
        id_node.appendChild(id_text)

        description_node = root.createElement('Description')
        rxn_node.appendChild(description_node)
        description_text = root.createTextNode(str(rxn))
        description_node.appendChild(description_text)

        type_node = root.createElement('Type')
        rxn_node.appendChild(type_node)
        type_text = root.createTextNode('mass-action')
        type_node.appendChild(type_text)

        rate_node = root.createElement('Rate')
        rxn_node.appendChild(rate_node)
        rate_text = root.createTextNode(f'{rxn.rate_constant_stochastic}')
        rate_node.appendChild(rate_text)

        # reactants
        reactants_node = root.createElement('Reactants')
        rxn_node.appendChild(reactants_node)
        if rxn.is_bimolecular() and rxn.symmetric():
            reactant = rxn.reactants.species[0]
            reactant_node = root.createElement('SpeciesReference')
            reactants_node.appendChild(reactant_node)
            reactant_node.setAttribute('id', reactant.name)
            reactant_node.setAttribute('stoichiometry', '2')
        else:
            for reactant in rxn.reactants.species:
                reactant_node = root.createElement('SpeciesReference')
                reactants_node.appendChild(reactant_node)
                reactant_node.setAttribute('id', reactant.name)
                reactant_node.setAttribute('stoichiometry', '1')

        # products
        products_node = root.createElement('Products')
        rxn_node.appendChild(products_node)
        if rxn.num_products() == 2 and rxn.symmetric_products():
            product = rxn.reactants.species[0]
            product_node = root.createElement('SpeciesReference')
            products_node.appendChild(product_node)
            product_node.setAttribute('id', product.name)
            product_node.setAttribute('stoichiometry', '2')
        else:
            for product in rxn.products.species:
                product_node = root.createElement('SpeciesReference')
                products_node.appendChild(product_node)
                product_node.setAttribute('id', product.name)
                product_node.setAttribute('stoichiometry', '1')

    stochkit_xml = root.toprettyxml(indent='  ')

    return stochkit_xml

def write_stochkit_file(filename: str, rxns: Iterable[Reaction], init_config: Dict[Specie, int], name: str = 'CRN') -> None:
    """
    Write stochkit file
    Args:
        rxns: reactions to translate to StochKit format
        init_config: dict mapping each :any:`Specie` to its initial count
        name: name of the CRN

    """
    xml = stochkit_format(rxns, init_config, name)
    with open(filename, 'w') as f:
        f.write(xml)

def species(species_: str) -> Union[Specie, Tuple[Specie]]:
    """
    Create a list of :any:`Specie` (Single species :any:`Expression`'s),
    or a single one.
    args:
        species_: str
            A space-separated string representing the names of the species
            being created
    This is normally used like this:
        w, x, y, z = species("W X Y Z")
        rxn = x + y >> z + w
        ...
    The names MUST be valid Python identifiers: "X0" is valid but "0X" is not.
    """
    species_list = species_.split()

    if len(species_list) == 1:
        return Specie(species_list[0])
    if len(species_list) != len(set(species_list)):
        raise ValueError(f'species_list {species_list} cannot contain duplicates.')

    return tuple(map(Specie, species_list))


SpeciePair = Tuple['Specie', 'Specie']  # forward annotations don't seem to work here
Output = Union[SpeciePair, Dict[SpeciePair, float]]


def replace_reversible_rxns(rxns: Iterable[Reaction]) -> List[Reaction]:
    """
    Args:
        rxns: list of :any:`Reaction`'s

    Returns: list of :any:`Reaction`'s, where every reversible reaction in `rxns` has been replaced by
        two irreversible reactions, and all others have been left as they are
    """
    new_rxns: List[Reaction] = []
    for rxn in rxns:
        if not rxn.reversible:
            new_rxns.append(rxn)
        else:
            forward_rxn = Reaction(reactants=rxn.reactants, products=rxn.products, k=rxn.rate_constant,
                                   rate_constant_units=rxn.rate_constant_units, reversible=False)
            reverse_rxn = Reaction(reactants=rxn.products, products=rxn.reactants, k=rxn.rate_constant_reverse,
                                   rate_constant_units=rxn.rate_constant_reverse_units, reversible=False)
            new_rxns.extend([forward_rxn, reverse_rxn])
    return new_rxns


# XXX: This algorithm currently uses the reactant *ordered* pair.
# We should think about the logic of that and see if it makes sense to collapse
# two reversed ordered pairs to a single unordered pair at this step,
# or whether that should be done explicitly by the user specifying transition_order='symmetric'.
def reactions_to_dict(reactions: Iterable[Reaction], n: int, volume: float) \
        -> Tuple[Dict[SpeciePair, Output], float]:
    """
    Returns dict representation of `reactions`, transforming unimolecular reactions to bimolecular,
    and converting rates to probabilities, also returning the max rate so the :any:`Simulator` knows
    how to scale time.

    Args:
        reactions: list of :any:`Reaction`'s
        n: the population size, necessary for rate conversion
        volume: parameter as defined in Gillespie algorithm

    Returns:
        (transitions_dict, max_rate), where `transitions_dict` is the dict representation of the transitions,
        and `max_rate` is the maximum rate for any pair of reactants,
        i.e., if we have reactions (a + b >> c + d).k(2) and (a + b >> x + y).k(3),
        then the ordered pair (a,b) has rate 2+3 = 5
    """
    reactions = list(copy.deepcopy(reactions))

    # add reverse reactions explicitly
    reversible_reactions = [reaction for reaction in reactions if reaction.reversible]
    for reaction in reversible_reactions:
        reversed_reaction = Reaction(reactants=reaction.products,
                                     products=reaction.reactants,
                                     k=reaction.rate_constant_reverse,
                                     rate_constant_units=reaction.rate_constant_reverse_units,
                                     reversible=False)
        reactions.append(reversed_reaction)
        reaction.reversible = False

    # Make a copy of reactions because this conversion will mutate the reactions
    reactions = convert_unimolecular_to_bimolecular_and_flip_reactant_order(reactions, n, volume)

    # for each ordered pair of reactants, calculate sum of rate constants across all reactions with those reactants
    # get in stochastic units no matter how they're encoded in the reactions
    reactant_pair_rates: DefaultDict[SpeciePair, float] = defaultdict(int)
    for reaction in reactions:
        if not reaction.is_bimolecular():
            raise ValueError(f'all reactions must have exactly two reactants, violated by {reaction}')
        if not reaction.num_products() == 2:
            raise ValueError(f'all reactions must have exactly two products, violated by {reaction}')
        reactants = reaction.reactants_if_bimolecular()
        reactant_pair_rates[reactants] += reaction.rate_constant_stochastic

    # divide all rate constants by the max rate (per reactant pair) to help turn them into probabilities
    max_rate = max(reactant_pair_rates.values())

    # add one randomized transition per reaction
    transitions: Dict[SpeciePair, Output] = {}
    for reaction in reactions:
        prob = reaction.rate_constant_stochastic / max_rate
        reactants = reaction.reactants_if_bimolecular()
        products = reaction.products_if_exactly_two()
        if reactants not in transitions:
            # should we be worried about floating-point error here?
            # I hope not since dividing a float by itself should always result in 1.0.
            transitions[reactants] = products if prob == 1.0 else {products: prob}
        else:
            # if we calculated probabilities correctly above, if we assigned a dict entry to be non-randomized
            # (since prob == 1.0 above), we should not encounter another reaction with same reactants
            assert (isinstance(transitions[reactants], dict))
            transitions[reactants][products] = prob

    # assert that each possible input for transitions has output probabilities summing to 1
    for reactants, outputs in transitions.items():
        if isinstance(outputs, dict):
            sum_probs = sum(prob for prob in outputs.values())
            assert sum_probs <= 1.0, f'sum_probs exceeds 1: {sum_probs}'

    return transitions, max_rate


def convert_unimolecular_to_bimolecular_and_flip_reactant_order(reactions: Iterable[Reaction], n: int, volume: float) \
        -> List[Reaction]:
    """Process all reactions before being added to the dictionary.

    bimolecular reactions have their rates multiplied by the corrective factor (n-1) / (2 * volume).
    Bimolecular reactions with two different reactants are added twice, with their reactants in both orders.
    """

    # gather set of all species together
    all_species: Set[Specie] = set()
    for reaction in reactions:
        all_species.update(reaction.get_species())

    converted_reactions = []
    for reaction in reactions:
        if reaction.num_reactants() != reaction.num_products():
            raise ValueError(
                f'each reaction must have same number of reactants and products, violated by {reaction}')
        if reaction.is_bimolecular():
            # Corrective factor to reaction rate
            reaction.rate_constant *= (n - 1) / (2 * volume)
            converted_reactions.append(reaction)
            # Add a flipped copy if the reaction has two different reactants
            reactants = reaction.reactants_if_bimolecular()
            if len(set(reactants)) > 1:
                flipped_reaction = copy.copy(reaction)
                flipped_reaction.reactants = Expression([reactants[1], reactants[0]])
                assert flipped_reaction.reactants_if_bimolecular != reactants
                converted_reactions.append(flipped_reaction)

        elif reaction.is_unimolecular():
            # for each unimolecular reaction R -->(k) P and each species S,
            # add a bimolecular reaction R+S -->(k) P+S
            reactant = reaction.reactant_if_unimolecular()
            product = reaction.product_if_unique()
            bimolecular_implementing_reactions = [(reactant + s >> product + s).k(reaction.rate_constant)
                                                  for s in all_species]

            converted_reactions.extend(bimolecular_implementing_reactions)
        else:
            raise ValueError(f'each reaction must have exactly one or two reactants, violated by {reaction}')
    return converted_reactions


@dataclass(frozen=True)
class Specie:
    name: str

    def __add__(self, other):
        if type(other) is Expression:
            return other + Expression([self])
        elif type(other) is Specie:
            return Expression([self]) + Expression([other])

        return NotImplemented

    __radd__ = __add__

    def __rshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other)

    def __rrshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(other, self)

    def __or__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __mul__(self, other: int) -> Expression:
        if type(other) is int:
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __rmul__(self, other: int) -> Expression:
        if type(other) is int:
            return other * Expression([self])
        else:
            raise NotImplementedError()

    def __str__(self):
        return self.name

    def __lt__(self, other: Specie) -> bool:
        return self.name < other.name

    def __eq__(self, other: Specie) -> bool:
        return self.name == other.name

    __req__ = __eq__


@dataclass(frozen=True)
class Expression:
    """
    Class used for very basic symbolic manipulation of left/right hand
    side of stoichiometric equations. Not very user friendly; users should
    just use the `species` functions and manipulate those to get their
    reactions.
    args:
        species: List[Specie]
            represents species in this expression
    properties:
        species: Dict[Specie, int]
            represents species and their coefficients (ints)
            all added together. The same as the argument passed to the
            constructor
    """

    species: List[Specie]

    def __add__(self, other: Expression) -> Expression:
        if type(other) is Expression:
            species_copy = list(self.species)
            species_copy.extend(other.species)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    def __rmul__(self, coeff: int) -> Expression:
        if type(coeff) is int:
            species_copy = []
            for _ in range(coeff):
                species_copy.extend(self.species)
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    __mul__ = __rmul__

    def __rshift__(self, expr: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, expr)

    def __or__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other, reversible=True)

    def __str__(self) -> str:
        return ' + '.join(s.name for s in self.species)

    def __len__(self) -> int:
        return len(self.species)

    def get_species(self) -> Set[Specie]:
        """
        Returns the set of species in this expression, not their
        coefficients.
        """
        return set(self.species)


avogadro = 6.02214076e23


def concentration_to_count(concentration: float, volume: float) -> int:
    """

    Args:
        concentration: units of M (molar) = moles / liter
        volume: units of liter

    Returns:
        count of molecule with `concentration` in `volume`
    """
    return round(avogadro * concentration * volume)


class RateConstantUnits(Enum):
    stochastic = 'stochastic'
    """Units of L/s. Multiple by Avogadro's number to convert to mass-action units."""

    mass_action = 'mass_action'
    """Units of /M/s. Divide by Avogadro's number to convert to stochastic units."""


@dataclass
class Reaction:
    """
    Representation of a stoichiometric reaction using a pair of Expressions,
    one for the reactants and one for the products.
    args:
        reactants: Expression
            The left hand side of the stoichiometric equation
        products: Expression
            The right hand side of the stoichiometric equation
        k: float
            The rate constant of the reaction
    properties:
        reactants: Expression
            The left hand side of the stoichiometric equation
        products: Expression
            The right hand side of the stoichiometric equation
        rate_constant: float
            The rate constant of the reaction
    """

    reactants: Expression
    products: Expression
    rate_constant: float = 1
    rate_constant_reverse: float = 1
    rate_constant_units: RateConstantUnits = RateConstantUnits.stochastic
    rate_constant_reverse_units: RateConstantUnits = RateConstantUnits.stochastic
    reversible: bool = False

    def __init__(self, reactants: Union[Specie, Expression], products: Union[Specie, Expression],
                 k: float = 1, r: float = 1,
                 rate_constant_units: RateConstantUnits = RateConstantUnits.stochastic,
                 rate_constant_reverse_units: RateConstantUnits = RateConstantUnits.stochastic,
                 reversible: bool = False) -> None:
        if type(reactants) not in (Specie, Expression):
            raise ValueError(
                "Attempted construction of reaction with type of reactants "
                f"as {type(reactants)}. Type of reactants must be Species "
                "or Expression")
        if type(products) not in (Specie, Expression):
            raise ValueError(
                "Attempted construction of products with type of products "
                f"as {type(products)}. Type of products must be Species "
                "or Expression")

        if type(reactants) is Specie:
            reactants = Expression([reactants])
        if type(products) is Specie:
            products = Expression([products])

        self.reactants = reactants
        self.products = products
        self.rate_constant = float(k)
        self.rate_constant_reverse = float(r)
        self.rate_constant_units = rate_constant_units
        self.rate_constant_reverse_units = rate_constant_reverse_units
        self.reversible = reversible

    def is_unimolecular(self) -> bool:
        return self.num_reactants() == 1

    def is_bimolecular(self) -> bool:
        return self.num_reactants() == 2

    def symmetric(self) -> bool:
        return self.num_reactants() == 2 and self.reactants.species[0] == self.reactants.species[1]

    def symmetric_products(self) -> bool:
        return self.num_products() == 2 and self.products.species[0] == self.products.species[1]

    def num_reactants(self) -> int:
        return len(self.reactants)

    def num_products(self) -> int:
        return len(self.products)

    def is_conservative(self) -> bool:
        return self.num_reactants() == self.num_products()

    def reactant_if_unimolecular(self) -> Specie:
        if self.is_unimolecular():
            return self.reactants.species[0]
        else:
            raise ValueError(f'reaction {self} is not unimolecular')

    def product_if_unique(self) -> Specie:
        if self.num_products() == 1:
            return self.products.species[0]
        else:
            raise ValueError(f'reaction {self} does not have exactly one product')

    def reactants_if_bimolecular(self) -> Tuple[Specie, Specie]:
        if self.is_bimolecular():
            return self.reactants.species[0], self.reactants.species[1]
        else:
            raise ValueError(f'reaction {self} is not bimolecular')

    def reactant_names_if_bimolecular(self) -> Tuple[str, str]:
        r1, r2 = self.reactants_if_bimolecular()
        return r1.name, r2.name

    def products_if_exactly_two(self) -> Tuple[Specie, Specie]:
        if self.num_products() == 2:
            return self.products.species[0], self.products.species[1]
        else:
            raise ValueError(f'reaction {self} does not have exactly two products')

    def product_names_if_exactly_two(self) -> Tuple[str, str]:
        p1, p2 = self.products_if_exactly_two()
        return p1.name, p2.name

    # def _get_exactly_two_species_from_dict(self, species_dict: Dict[Specie, int]) -> Tuple[Specie, Specie]:
    #     if len(species_dict) == 1:
    #         # stoichiometric coefficient 2
    #         specie1 = specie2 = next(iter(species_dict.keys()))
    #         specie_tuple = (specie1, specie2)
    #     else:
    #         specie_tuple = tuple(species_dict.keys())
    #     return specie_tuple

    def __str__(self):
        rev_rate_str = f'({self.rate_constant_reverse})<' if self.reversible else ''
        return f"{self.reactants} {rev_rate_str}-->({self.rate_constant}) {self.products}"

    def __repr__(self):
        return (f"Reaction({repr(self.reactants)}, {repr(self.products)}, "
                f"{self.rate_constant})")

    @property
    def rate_constant_stochastic(self) -> float:
        return self.rate_constant \
            if self.rate_constant_units == RateConstantUnits.stochastic \
            else self.rate_constant / avogadro

    @property
    def rate_constant_reverse_stochastic(self) -> float:
        return self.rate_constant_reverse \
            if self.rate_constant_reverse_units == RateConstantUnits.stochastic \
            else self.rate_constant_reverse / avogadro

    def k(self, coeff: float, units: RateConstantUnits = RateConstantUnits.stochastic) -> Reaction:
        """
        Changes the reaction coefficient to `coeff` and returns `self`.

        args:
            coeff: float
                The new reaction coefficient
            units: float
                units of rate constant (default stochastic)

        This is useful for including the rate constant during the construction
        of a reaction. For example
            x, y, z = species("X Y Z")
            sys = CRN(
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5))
            ...
        """
        if self.is_unimolecular() and units == RateConstantUnits.mass_action:
            raise ValueError('cannot use mass-action rate constants on a unimolecular reaction')
        self.rate_constant_units = units
        self.rate_constant = coeff
        return self

    def r(self, coeff: float, units: RateConstantUnits = RateConstantUnits.stochastic) -> Reaction:
        """
        Changes the reverse reactionn reaction rate constant to `coeff` and returns `self`.

        args:
            coeff: float
                The new reverse reaction rate constant
            units: float
                units of rate constant (default stochastic)

        This is useful for including the rate constant during the construction
        of a reaction. For example
            x, y, z = species("X Y Z")
            sys = CRN(
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5))
            ...
        """
        if self.num_products() == 1 and units == RateConstantUnits.mass_action:
            raise ValueError('cannot use mass-action rate constants on a unimolecular reaction; '
                             'this reaction has only one product, so its reverse is unimolecular')
        if not self.reversible:
            raise ValueError('cannot set r on an irreversible reaction')
        self.rate_constant_reverse_units = units
        self.rate_constant_reverse = coeff
        return self

    def get_species(self):
        """
        Returns the set of species present in the products and reactants.
        """
        return {
            *self.reactants.get_species(),
            *self.products.get_species()
        }
