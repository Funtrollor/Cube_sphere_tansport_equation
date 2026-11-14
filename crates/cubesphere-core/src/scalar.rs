//! Scalar operations shared by `f32` and `f64` runtime field buffers.
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Floating-point scalar abstraction allowing identical solver logic in `f32` and `f64`.
///
/// Precomputation (LGL nodes/weights, geometry, metrics) is performed in `f64` for maximal
/// accuracy and converted to `Self` when building runtime field buffers.
pub trait FloatScalar:
    Copy
    + Clone
    + Debug
    + Display
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Additive identity, `0`.
    const ZERO: Self;
    /// Multiplicative identity, `1`.
    const ONE: Self;
    /// The scalar value `0.5`, used for split-form averages.
    const HALF: Self;
    /// The scalar value `2`.
    const TWO: Self;
    /// Pi at the implementing scalar's precision, for angular calculations in radians.
    const PI: Self;

    /// Converts an `f64` precomputed value; `f32` uses a narrowing cast.
    fn from_f64(v: f64) -> Self;
    /// Converts to `f64` for diagnostics; widening cannot recover lost `f32` precision.
    fn to_f64(self) -> f64;
    /// Absolute value using the underlying floating-point operation.
    fn abs(self) -> Self;
    /// Principal square root; negative inputs produce NaN.
    fn sqrt(self) -> Self;
    /// Sine of an angle in radians.
    fn sin(self) -> Self;
    /// Cosine of an angle in radians.
    fn cos(self) -> Self;
    /// Tangent of an angle in radians, used by equiangular projection.
    fn tan(self) -> Self;
    /// Principal arcsine in radians, in `[-pi/2, pi/2]` for inputs in `[-1, 1]`.
    fn asin(self) -> Self;
    /// Principal arccosine in radians, in `[0, pi]` for inputs in `[-1, 1]`.
    fn acos(self) -> Self;
    /// Four-quadrant angle in radians, treating `self` as `y` and `other` as `x`.
    fn atan2(self, other: Self) -> Self;
    /// Natural exponential `e^self`.
    fn exp(self) -> Self;
    /// Raises `self` to the floating-point power `n`.
    fn powf(self, n: Self) -> Self;
    /// Smaller operand, returning the non-NaN operand if only one is NaN.
    fn min(self, other: Self) -> Self;
    /// Larger operand, returning the non-NaN operand if only one is NaN.
    fn max(self, other: Self) -> Self;
    /// Restricts the value to the inclusive interval `[min, max]`, preserving a NaN input.
    /// # Panics
    /// The `f32` and `f64` implementations panic if `min > max` or either bound is NaN.
    fn clamp_val(self, min: Self, max: Self) -> Self;
}

impl FloatScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const PI: Self = std::f64::consts::PI;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn clamp_val(self, min: Self, max: Self) -> Self {
        f64::clamp(self, min, max)
    }
}

impl FloatScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const PI: Self = std::f32::consts::PI;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f32::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        f32::atan2(self, other)
    }
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn clamp_val(self, min: Self, max: Self) -> Self {
        f32::clamp(self, min, max)
    }
}
