use pyo3::prelude::*;

mod simulator;
mod urn;

use urn::Urn;
use simulator::SimulatorSequentialArray;
use simulator::SimulatorMultiBatch;

/// A Python module implemented in Rust using PyO3.
#[pymodule]
fn simulator_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add all classes
    m.add_class::<Urn>()?;
    m.add_class::<SimulatorSequentialArray>()?;
    m.add_class::<SimulatorMultiBatch>()?;
    
    Ok(())
}
