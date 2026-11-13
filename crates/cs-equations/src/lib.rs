pub mod advection;
pub mod initial;
pub mod wind;

pub use advection::{Equation, ScalarAdvection};
pub use initial::{ConstantField, CosineBell, GaussianBell, InitialKind, ScalarField, TwinGaussianBells};
pub use wind::{NairLauritzenFlow, SolidBodyRotation, WindDescriptor, WindField, WindKind};
