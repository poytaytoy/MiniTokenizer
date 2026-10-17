mod splice;
mod wordarena;

const REGEX_GPT_2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

mod train;

mod tokenizer;
pub use tokenizer::MiniTokenizer;

#[cfg(feature = "python")]
mod python_bindings {
    use super::MiniTokenizer;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    /// Train, load, encode, and decode a BPE model.
    /// Keep each instance on the thread that created it.
    #[pyclass(name = "MiniTokenizer", module = "tokenizer", unsendable)]
    struct PythonMiniTokenizer {
        inner: MiniTokenizer,
    }

    #[pymethods]
    impl PythonMiniTokenizer {
        /// Load a model from a JSON string.
        #[staticmethod]
        fn load(model_json: &str) -> PyResult<Self> {
            MiniTokenizer::load(model_json)
                .map(|inner| Self { inner })
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        }

        /// Learn from a text string and return a ready tokenizer.
        #[staticmethod]
        fn train(text: &str, vocab_size: usize) -> PyResult<Self> {
            MiniTokenizer::train(text, vocab_size)
                .map(|inner| Self { inner })
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        }

        /// Export the model as a JSON string.
        fn to_json(&self) -> String {
            self.inner.to_json().to_owned()
        }

        /// Convert text into a list of token IDs.
        fn encode(&mut self, text: &str) -> PyResult<Vec<u32>> {
            // Report missing vocabulary as an ordinary Python exception.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.encode(text)))
                .map_err(|_| {
                    PyValueError::new_err(
                        "Text contains a character or token missing from the model",
                    )
                })
        }

        /// Convert token IDs back into text. Unknown IDs raise ValueError.
        fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
            self.inner
                .decode(&ids)
                .map_err(|err| PyValueError::new_err(err.to_string()))
        }
    }

    #[pymodule]
    fn tokenizer(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<PythonMiniTokenizer>()
    }
}
