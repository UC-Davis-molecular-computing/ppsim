# Rust Implementation for ppsim

This directory contains a Rust implementation of the simulator for the ppsim package. The Rust implementation is designed to be a drop-in replacement for the existing Cython implementation, providing better performance while maintaining the same API.

## Building the Rust Extension

To build the Rust extension, you need to have Rust and Cargo installed. You can install them using [rustup](https://rustup.rs/).

Once you have Rust and Cargo installed, you can build the Rust extension using maturin:

```bash
pip install maturin
maturin develop
```

This will build the Rust extension and install it in your current Python environment.

## Using the Rust Implementation

The ppsim package provides a mechanism to switch between the Cython and Rust implementations at runtime. You can use the `use_rust_simulator` function to switch between them:

```python
import ppsim

# Use the Rust implementation
ppsim.use_rust_simulator(True)

# Use the Cython implementation
ppsim.use_rust_simulator(False)

# Check which implementation is being used
print(ppsim.get_simulator_implementation())
```

## Testing the Implementations

You can use the `test_simulator_implementations.py` script to test both implementations and compare their performance:

```bash
python test_simulator_implementations.py
```

This script will run a simple simulation with both implementations and compare their performance.

## Implementation Details

The Rust implementation is located in the `ppsim/rust` directory. It consists of the following files:

- `src/lib.rs`: The main entry point for the Rust extension.
- `src/simulator.rs`: The implementation of the simulator classes.
- `src/urn.rs`: The implementation of the Urn class.

The Rust implementation is designed to be a drop-in replacement for the Cython implementation, so it provides the same API and behavior.

## Packaging

When packaging the ppsim package for distribution, you can include both the Cython and Rust implementations. The package will use the Cython implementation by default, but users can switch to the Rust implementation if it's available.

To build a wheel that includes both implementations, you can use maturin:

```bash
maturin build
```

This will create a wheel file in the `target/wheels` directory that includes both the Cython and Rust implementations.
