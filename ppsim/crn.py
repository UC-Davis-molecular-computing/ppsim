"""
Module for expression population protocols using CRN notation. Ideas and much code taken from
https://github.com/enricozb/python-crn.
"""

from __future__ import annotations  # needed for forward references in type hints

from collections import defaultdict
from typing import Union, Dict, Tuple, Set, Iterable, DefaultDict, List
from dataclasses import dataclass
from functools import reduce
from operator import mul


def species(species_: str) -> Union[Specie, Tuple[Specie]]:
    """
    Create a list of :any:`Specie` (Single species :any:`Expression`'s),
    or a single one.
    args:
        species_: str
            A space-seperated string representing the names of the species
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

    return tuple(map(Specie, species_list))


SpeciePair = Tuple['Specie', 'Specie']  # forward annotations don't seem to work here
Output = Union[SpeciePair, Dict[SpeciePair, float]]


# XXX: This algorithm currently uses the reactant *ordered* pair.
# We should think about the logic of that and see if it makes sense to collapse
# two reversed ordered pairs to a single unordered pair at this step,
# or whether that should be done explicitly by the user specifying transition_order='symmetric'.
def reactions_to_dict(reactions: Iterable[Reaction]) -> Tuple[Dict[SpeciePair, Output], float]:
    """
    Returns dict representation of `reactions`, transforming unimolecular reactions to bimolecular,
    and converting rates to probabilities, also returning the max rate so the :any:`Simulator` knows
    how to scale time.

    Args:
        reactions: list of :any:`Reaction`'s

    Returns:
        (transitions_dict, max_rate), where `transitions_dict` is the dict representation of the transitions,
        and `max_rate` is the maximum rate for any pair of reactants,
        i.e., if we have reactions (a + b >> c + d).k(2) and (a + b >> x + y).k(3),
        then the ordered pair (a,b) has rate 2+3 = 5
    """
    reactions = convert_unimolecular_to_bimolecular(reactions)

    # for each ordered pair of reactants, calculate sum of rate constants across all reactions with those reactants
    reactant_pair_rates: DefaultDict[SpeciePair, float] = defaultdict(int)
    for reaction in reactions:
        if not reaction.is_bimolecular():
            raise ValueError(f'all reactions must have exactly two reactants, violated by {reaction}')
        if not reaction.num_products() == 2:
            raise ValueError(f'all reactions must have exactly two products, violated by {reaction}')
        reactants = reaction.reactants_if_bimolecular()
        reactant_pair_rates[reactants] += reaction.rate_constant

    # divide all rate constants by the max rate (per reactant pair) to help turn them into probabilities
    max_rate = max(reactant_pair_rates.values())

    # add one randomized transition per reaction
    transitions: Dict[SpeciePair, Output] = {}
    for reaction in reactions:
        prob = reaction.rate_constant / max_rate
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


def convert_unimolecular_to_bimolecular(reactions: Iterable[Reaction]) -> List[Reaction]:
    # gather set of all species together
    all_species: Set[Specie] = set()
    for reaction in reactions:
        all_species.update(reaction.get_species())

    converted_reactions = []
    for reaction in reactions:
        if reaction.num_reactants() != reaction.num_products():
            raise ValueError(f'each reaction must have same number of reactants and products, violated by {reaction}')
        if reaction.is_bimolecular():
            # pass along existing bimolecular reactions unchanged
            converted_reactions.append(reaction)
        elif reaction.is_unimolecular():
            # for each unimolecular reaction R -->(k) P and each species S,
            # add a bimolecular reaction R+S -->(k) P+S
            reactant = reaction.reactant_if_unimolecular()
            product = reaction.product_if_unique()
            bimolecular_implementing_reactions = [(reactant + s >> product + s).k(reaction.rate_constant) for s in
                                                  all_species]
            converted_reactions.extend(bimolecular_implementing_reactions)
        else:
            raise ValueError(f'each reaction must have exactly one or two reactants, violated by {reaction}')
    return converted_reactions


@dataclass(frozen=True)
class Specie:
    name: str

    def __add__(self, other):
        if type(other) is Expression:
            return other + Expression({self: 1})
        elif type(other) is Specie:
            return Expression({self: 1}) + Expression({other: 1})

        return NotImplemented

    __radd__ = __add__

    def __rshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, other)

    def __rrshift__(self, other: Union[Specie, Expression]) -> Reaction:
        return Reaction(other, self)

    def __mul__(self, other: int) -> Expression:
        if type(other) is int:
            return Expression({self: other})
        else:
            raise NotImplementedError()

    def __rmul__(self, other: int) -> Expression:
        if type(other) is int:
            return Expression({self: other})
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
        species: Dict[Specie, int]
            represents species and their coefficients (ints)
            all added together.
    properties:
        species: Dict[Specie, int]
            represents species and their coefficients (ints)
            all added together. The same as the argument passed to the
            constructor
    """

    species: Dict[Specie, int]

    def __add__(self, other: Expression) -> Expression:
        if type(other) is Expression:
            species_copy = self.species.copy()
            for s, c in other.species.items():
                if s not in species_copy:
                    species_copy[s] = 0
                species_copy[s] += c
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    def __rmul__(self, coeff: int) -> Expression:
        if type(coeff) is int:
            species_copy = {}
            for s, c in self.species.items():
                species_copy[s] = c * coeff
            return Expression(species_copy)
        else:
            raise NotImplementedError()

    __mul__ = __rmul__

    def __rshift__(self, expr: Union[Specie, Expression]) -> Reaction:
        return Reaction(self, expr)

    def __str__(self) -> str:
        return ' + '.join(
            map(lambda i: f"{i[1] if i[1] != 1 else ''}{i[0]}",
                self.species.items()))

    def __repr__(self) -> str:
        return ' + '.join(
            map(lambda i: f"{i[1] if i[1] != 1 else ''}{i[0]}",
                self.species.items()))

    def __len__(self) -> int:
        return sum(self.species.values())

    def get_species(self) -> Set[Specie]:
        """
        Returns the set of species in this expression, not their
        coefficients.
        """
        return set(self.species.keys())


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
        coeff: float
            The rate constant of the reaction
    """

    reactants: Expression
    products: Expression
    rate_constant: float

    def __init__(self, reactants: Union[Specie, Expression], products: Union[Specie, Expression], k: float = 1) -> None:
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
            reactants = Expression({reactants: 1})
        if type(products) is Specie:
            products = Expression({products: 1})

        self.reactants = reactants
        self.products = products
        self.rate_constant = float(k)

    def is_unimolecular(self) -> bool:
        return self.num_reactants() == 1

    def is_bimolecular(self) -> bool:
        return self.num_reactants() == 2

    def num_reactants(self) -> int:
        return len(self.reactants)

    def num_products(self) -> int:
        return len(self.products)

    def is_conservative(self) -> bool:
        return self.num_reactants() == self.num_products()

    def reactant_if_unimolecular(self) -> Specie:
        if self.is_unimolecular():
            return next(iter(self.reactants.species.keys()))
        else:
            raise ValueError(f'reaction {self} is not unimolecular')

    def product_if_unique(self) -> Specie:
        if self.num_products() == 1:
            return next(iter(self.products.species.keys()))
        else:
            raise ValueError(f'reaction {self} does not have exactly one product')

    def reactants_if_bimolecular(self) -> Tuple[Specie, Specie]:
        if self.is_bimolecular():
            return self._get_exactly_two_species_from_dict(self.reactants.species)
        else:
            raise ValueError(f'reaction {self} is not bimolecular')

    def reactant_names_if_bimolecular(self) -> Tuple[str, str]:
        r1, r2 = self.reactants_if_bimolecular()
        return r1.name, r2.name

    def products_if_exactly_two(self) -> Tuple[Specie, Specie]:
        if self.num_products() == 2:
            return self._get_exactly_two_species_from_dict(self.products.species)
        else:
            raise ValueError(f'reaction {self} does not have exactly two products')

    def product_names_if_exactly_two(self) -> Tuple[str, str]:
        p1, p2 = self.products_if_exactly_two()
        return p1.name, p2.name

    def _get_exactly_two_species_from_dict(self, species_dict: Dict[Specie, int]) -> Tuple[Specie, Specie]:
        if len(species_dict) == 1:
            # stoichiometric coefficient 2
            specie1 = specie2 = next(iter(species_dict.keys()))
            specie_tuple = (specie1, specie2)
        else:
            specie_tuple = tuple(species_dict.keys())
        return specie_tuple

    def __str__(self):
        return f"{self.reactants} -->({self.rate_constant}) {self.products}"
        # rcts_str = str(self.reactants)
        # return (f"{' ' * len(rcts_str)} {self.coeff:.1f} \n"
        #         f"{self.reactants} ---> {self.products}")

    def __repr__(self):
        return (f"Reaction({repr(self.reactants)}, {repr(self.products)}, "
                f"{self.rate_constant})")

    def k(self, coeff: float) -> Reaction:
        """
        Changes the reaction coefficient to `coeff` and returns `self`.
        args:
            coeff: float
                The new reaction coefficient
        This is useful for including the rate constant during the construction
        of a reaction. For example
            x, y, z = species("X Y Z")
            sys = CRN(
                (x + y >> z).k(2.5),
                (z >> x).k(1.5),
                (z >> y).k(0.5))
            ...
        """
        self.rate_constant = coeff
        return self

    def get_species(self):
        """
        Returns the set of species present in the products and reactants.
        """
        return {
            *self.reactants.get_species(),
            *self.products.get_species()
        }

    def net_production(self, species_):
        """
        Returns the net stoichiometric coefficient of a species in this
        reaction.
        args:
            species: str
                string name of the species
        """
        return (self.products.species.get(species_, 0) -
                self.reactants.species.get(species_, 0))

    def propensity(self, counts):
        """
        Returns the value of Reaction.discrete_flux given the currently
        present molecules/species in 'counts'.
        """

        def flux_part(i):
            s, c = i
            return reduce(mul, (counts.get(s, 0) - i for i in range(c)))

        return reduce(mul, map(flux_part, self.reactants.species.items()))
