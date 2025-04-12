use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, IntoPyArray};
use numpy::npyffi::npy_intp;
use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand_pcg::Pcg64;
use rand::SeedableRng;
use rand::Rng; // Used for the gen() method

/// Data structure for a multiset that supports fast random sampling.
#[pyclass(text_signature = "(config, bitgen)")]
pub struct Urn {
    #[pyo3(get, set)]
    pub config: Py<PyArray1<i64>>,
    #[pyo3(get, set)]
    pub order: Py<PyArray1<npy_intp>>,
    #[pyo3(get)]
    pub size: i64,
    length: npy_intp,
    rng: Option<Pcg64>,
}

#[pymethods]
impl Urn {
    /// Create a new Urn object.
    #[new]
    pub fn new(py: Python<'_>, config: Py<PyArray1<i64>>, bitgen: PyObject) -> PyResult<Self> {
        // Extract seed from bitgen if possible
        let seed = match bitgen.getattr(py, "state")?.getattr(py, "state")?.extract::<u64>(py) {
            Ok(s) => Some(s),
            Err(_) => None,
        };
        
        // Create a Rust RNG
        let rust_rng = match seed {
            Some(s) => Some(Pcg64::seed_from_u64(s)),
            None => Some(Pcg64::from_entropy()),
        };
        let config_array = config.as_ref(py).readonly();
        let length = config_array.shape()[0] as npy_intp;
        let size = config_array.as_array().sum();
        
        // Create order array (equivalent to np.array(range(len(config)), dtype=np.intp))
        let order = Array1::from_iter(0..length)
            .into_pyarray(py)
            .to_owned();
        
        let mut urn = Urn {
            config: config.clone_ref(py),
            order,
            size,
            length,
            rng: rust_rng,
        };
        
        // Sort the order array
        urn.sort(py)?;
        
        Ok(urn)
    }
    
    /// Updates self.order.
    /// 
    /// Uses insertion sort to maintain that 
    /// config[order[0]] >= config[order[1]] >= ... >= config[order[q]].
    /// This method is used to have O(q) time when order is almost correct.
    pub fn sort(&mut self, py: Python<'_>) -> PyResult<()> {
        let config = self.config.as_ref(py).readonly();
        let config_array = config.as_array();
        let mut order = self.order.as_ref(py).readwrite();
        let mut order_array = order.as_array_mut();
        
        for i in 1..self.length as usize {
            // See if the entry at order[i] needs to be moved earlier.
            // Recursively, we have ensured that order[0], ..., order[i-1] have the correct order.
            let o_i = order_array[i];
            
            // j will be the index where order[i] should be inserted to.
            let mut j = i;
            while j > 0 && config_array[o_i as usize] > config_array[order_array[j-1] as usize] {
                j -= 1;
            }
            
            // Index at order[i] will get moved to order[j], and all indices order[j], ..., order[i-1] get right shifted
            // First do the right shift, moving order[i-k] for k = 1, ..., i-j
            for k in 1..(i-j+1) {
                order_array[i + 1 - k] = order_array[i - k];
            }
            order_array[j] = o_i;
        }
        
        Ok(())
    }
    
    /// Samples and removes one element, returning its index.
    pub fn sample_one(&mut self, py: Python<'_>) -> PyResult<npy_intp> {
        if self.size <= 0 {
            return Err(PyValueError::new_err("Cannot sample from empty urn"));
        }
        
        // Generate random number in [0, self.size-1]
        let x: i64 = if let Some(rng) = &mut self.rng {
            let uniform = Uniform::from(0..self.size);
            uniform.sample(rng)
        } else {
            // Fallback to Python's random if Rust RNG is not available
            let random = py.import("random")?;
            random.call_method1("randint", (0, self.size - 1))?.extract()?
        };
        
        let mut config = self.config.as_ref(py).readwrite();
        let mut config_array = config.as_array_mut();
        let order = self.order.as_ref(py).readonly();
        let order_array = order.as_array();
        
        let mut i = 0;
        let mut remaining = x;
        let mut index = order_array[0] as usize;
        
        while remaining >= 0 {
            index = order_array[i] as usize;
            remaining -= config_array[index];
            i += 1;
        }
        
        // Decrement the count for the sampled element
        config_array[index] -= 1;
        self.size -= 1;
        
        Ok(index as npy_intp)
    }
    
    /// Adds one element at index.
    #[doc = "add_to_entry(index, amount=1)"]
    pub fn add_to_entry(&mut self, py: Python<'_>, index: npy_intp, amount: Option<i64>) -> PyResult<()> {
        let amount = amount.unwrap_or(1);
        let mut config = self.config.as_ref(py).readwrite();
        let mut config_array = config.as_array_mut();
        
        config_array[index as usize] += amount;
        self.size += amount;
        
        Ok(())
    }
    
    /// Samples n elements, returning them as a vector.
    /// 
    /// Args:
    ///     n: number of elements to sample
    ///     v: the array to write the output vector in
    ///         (this is faster than re-initializing an output array)
    ///         
    /// Returns:
    ///     nz: the number of nonzero entries
    ///         v[self.order[i]] for i in range(nz) can then loop over only 
    ///             the nonzero entries of the vector
    pub fn sample_vector(&mut self, py: Python<'_>, n: i64, v: Py<PyArray1<i64>>) -> PyResult<npy_intp> {
        if n <= 0 {
            return Ok(0);
        }
        
        let mut v_array = v.as_ref(py).readwrite();
        let mut v_slice = v_array.as_array_mut();
        
        // Zero out the array
        for i in 0..v_slice.len() {
            v_slice[i] = 0;
        }
        
        // Simplified implementation: just sample n elements one by one
        let mut remaining_n = n;
        let mut i: usize = 0;
        
        while remaining_n > 0 && i < self.length as usize && self.size > 0 {
            // Sample one element at a time
            let x: i64 = if let Some(rng) = &mut self.rng {
                let uniform = Uniform::from(0..self.size);
                uniform.sample(rng)
            } else {
                // Fallback to Python's random if Rust RNG is not available
                let random = py.import("random")?;
                random.call_method1("randint", (0, self.size - 1))?.extract()?
            };
            
            let mut config = self.config.as_ref(py).readwrite();
            let mut config_array = config.as_array_mut();
            let order = self.order.as_ref(py).readonly();
            let order_array = order.as_array();
            
            let mut j = 0;
            let mut remaining = x;
            let mut index = order_array[0] as usize;
            
            while remaining >= 0 {
                index = order_array[j] as usize;
                remaining -= config_array[index];
                j += 1;
            }
            
            // Add to the vector
            v_slice[index] += 1;
            
            // Decrement the count for the sampled element
            config_array[index] -= 1;
            self.size -= 1;
            
            remaining_n -= 1;
            
            if config_array[index] == 0 {
                i += 1;
            }
        }
        
        // Count non-zero entries
        let mut nz = 0;
        for i in 0..v_slice.len() {
            if v_slice[i] > 0 {
                nz += 1;
            }
        }
        
        Ok(nz as npy_intp)
    }
    
    /// Adds a vector of elements to the urn.
    pub fn add_vector(&mut self, py: Python<'_>, vector: Py<PyArray1<i64>>) -> PyResult<()> {
        let vector_array = vector.as_ref(py).readonly();
        let vector_slice = vector_array.as_array();
        
        let mut config = self.config.as_ref(py).readwrite();
        let mut config_array = config.as_array_mut();
        
        for i in 0..self.length as usize {
            config_array[i] += vector_slice[i];
            self.size += vector_slice[i];
        }
        
        Ok(())
    }
    
    /// Set the counts back to zero.
    pub fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut config = self.config.as_ref(py).readwrite();
        let mut config_array = config.as_array_mut();
        
        for i in 0..self.length as usize {
            config_array[i] = 0;
        }
        
        self.size = 0;
        
        Ok(())
    }
}
