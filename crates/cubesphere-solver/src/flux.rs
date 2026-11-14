//! Numerical flux selection and parsing for scalar advection SBP-SAT edge coupling.
use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Interface numerical flux used by the SBP-SAT element coupling.
///
/// All variants belong to the one-parameter alpha-flux family
/// `F* = v_n {q} - (alpha / 2) |v_n| [[q]]`, which yields the SAT penalty
/// `-(1/2) (v_n - alpha |v_n|) (q_out - q_in) / w_edge` added to the RHS.
///
/// - `alpha = 1`: full upwind (energy dissipative at interfaces)
/// - `alpha = 0`: central (interface coupling is exactly energy neutral)
///
/// Every member of the family is discretely mass conservative, because the
/// `alpha |v_n|` contributions of the two sides of an interface cancel in the
/// global integral.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NumericalFlux {
    /// Full upwind coupling (`alpha = 1`), dissipating interface jumps.
    #[default]
    Upwind,
    /// Central coupling (`alpha = 0`), with no interface jump dissipation.
    Central,
    /// Jump-dissipation weight `alpha`; use [`Self::from_upwind_weight`] to enforce `[0, 1]`.
    Blended(f64),
}

impl NumericalFlux {
    /// Weight `alpha` of the `|v_n|` jump term (`1` = upwind, `0` = central).
    /// A directly constructed `Blended` value is returned without validation or clamping.
    pub fn upwind_weight(self) -> f64 {
        match self {
            Self::Upwind => 1.0,
            Self::Central => 0.0,
            Self::Blended(alpha) => alpha,
        }
    }

    /// Builds a flux from its `alpha` weight, mapping the endpoints to named variants.
    /// # Errors
    /// Returns an error if `alpha` is non-finite or outside the inclusive interval `[0, 1]`.
    pub fn from_upwind_weight(alpha: f64) -> Result<Self, String> {
        if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
            return Err(format!("flux weight alpha must lie in [0, 1], got {alpha}"));
        }
        Ok(if alpha == 1.0 {
            Self::Upwind
        } else if alpha == 0.0 {
            Self::Central
        } else {
            Self::Blended(alpha)
        })
    }
}

impl Display for NumericalFlux {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upwind => write!(f, "upwind"),
            Self::Central => write!(f, "central"),
            Self::Blended(alpha) => write!(f, "blended(alpha={alpha})"),
        }
    }
}

impl FromStr for NumericalFlux {
    type Err = String;

    /// Accepts `upwind`, `central`, or a numeric `alpha` in `[0, 1]`.
    /// Trims whitespace and matches names without ASCII case sensitivity.
    /// # Errors
    /// Returns an error for an unknown name, invalid number, or non-finite/out-of-range weight.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "upwind" => Ok(Self::Upwind),
            "central" => Ok(Self::Central),
            other => {
                let alpha: f64 = other.parse().map_err(|_| {
                    format!("unknown flux '{s}' (expected upwind, central, or alpha in [0, 1])")
                })?;
                Self::from_upwind_weight(alpha)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flux_parsing_and_weights() {
        assert_eq!(
            "upwind".parse::<NumericalFlux>().unwrap(),
            NumericalFlux::Upwind
        );
        assert_eq!(
            "Central".parse::<NumericalFlux>().unwrap(),
            NumericalFlux::Central
        );
        assert_eq!("1".parse::<NumericalFlux>().unwrap(), NumericalFlux::Upwind);
        assert_eq!(
            "0.0".parse::<NumericalFlux>().unwrap(),
            NumericalFlux::Central
        );
        assert_eq!(
            "0.25".parse::<NumericalFlux>().unwrap(),
            NumericalFlux::Blended(0.25)
        );
        assert!("1.5".parse::<NumericalFlux>().is_err());
        assert!("-0.1".parse::<NumericalFlux>().is_err());
        assert!("nan".parse::<NumericalFlux>().is_err());
        assert!("lax".parse::<NumericalFlux>().is_err());

        assert_eq!(NumericalFlux::default().upwind_weight(), 1.0);
        assert_eq!(NumericalFlux::Central.upwind_weight(), 0.0);
        assert_eq!(NumericalFlux::Blended(0.5).upwind_weight(), 0.5);
    }
}
