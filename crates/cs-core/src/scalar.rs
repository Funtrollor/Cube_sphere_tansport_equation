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
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    const TWO: Self;
    const PI: Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn exp(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
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
