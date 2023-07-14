#![allow(unused)]
pub mod mat;
pub mod nn;
pub mod act;
pub mod batch;

pub mod prelude {
    pub use crate::nn::*;
    pub use crate::mat::*;
    pub use crate::act::*;
    pub use crate::batch::*;
}
