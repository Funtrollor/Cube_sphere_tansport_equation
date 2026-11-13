use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Single row of a convergence study (`h`-refinement or `p`-refinement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceRow {
    pub k: usize,
    pub poly_order: usize,
    pub total_dofs: usize,
    pub dt: f64,
    pub steps: usize,
    pub elapsed_secs: f64,
    pub l1_rel: f64,
    pub l2_abs: f64,
    pub l2_rel: f64,
    pub order_l2: Option<f64>,
    pub l_inf: f64,
    pub order_linf: Option<f64>,
    pub mass_rel_err: f64,
}

/// Computes empirical spatial convergence rates between consecutive rows.
pub fn populate_convergence_orders(rows: &mut [ConvergenceRow], vary_by_k: bool) {
    for i in 0..rows.len() {
        if i == 0 {
            rows[i].order_l2 = None;
            rows[i].order_linf = None;
            continue;
        }
        let ratio = if vary_by_k {
            (rows[i].k as f64) / (rows[i - 1].k as f64)
        } else {
            (rows[i].poly_order as f64) / (rows[i - 1].poly_order as f64)
        };
        let log_ratio = ratio.ln();
        if log_ratio > 0.0 && rows[i].l2_rel > 0.0 && rows[i - 1].l2_rel > 0.0 {
            rows[i].order_l2 = Some((rows[i - 1].l2_rel / rows[i].l2_rel).ln() / log_ratio);
        }
        if log_ratio > 0.0 && rows[i].l_inf > 0.0 && rows[i - 1].l_inf > 0.0 {
            rows[i].order_linf = Some((rows[i - 1].l_inf / rows[i].l_inf).ln() / log_ratio);
        }
    }
}

/// Computes empirical temporal convergence rates between consecutive rows (`dt` refinement).
pub fn populate_temporal_convergence_orders(rows: &mut [ConvergenceRow]) {
    for i in 0..rows.len() {
        if i == 0 {
            rows[i].order_l2 = None;
            rows[i].order_linf = None;
            continue;
        }
        let ratio = rows[i - 1].dt / rows[i].dt;
        let log_ratio = ratio.ln();
        if log_ratio > 0.0 && rows[i].l2_rel > 0.0 && rows[i - 1].l2_rel > 0.0 {
            rows[i].order_l2 = Some((rows[i - 1].l2_rel / rows[i].l2_rel).ln() / log_ratio);
        }
        if log_ratio > 0.0 && rows[i].l_inf > 0.0 && rows[i - 1].l_inf > 0.0 {
            rows[i].order_linf = Some((rows[i - 1].l_inf / rows[i].l_inf).ln() / log_ratio);
        }
    }
}

/// Formats spatial convergence rows as an aligned ASCII table.
pub fn format_convergence_table(rows: &[ConvergenceRow]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:>3} | {:>3} | {:>7} | {:>5} | {:>11} | {:>6} | {:>11} | {:>6} | {:>11} | {:>8}",
        "K", "p", "DOFs", "Steps", "L2 Rel", "Order", "Linf", "Order", "Mass Err", "Time(s)"
    );
    let _ = writeln!(out, "{}", "-".repeat(100));

    for row in rows {
        let ord2 = row
            .order_l2
            .map(|v| format!("{v:>6.2}"))
            .unwrap_or_else(|| "     -".to_string());
        let ord_inf = row
            .order_linf
            .map(|v| format!("{v:>6.2}"))
            .unwrap_or_else(|| "     -".to_string());
        let _ = writeln!(
            out,
            "{:>3} | {:>3} | {:>7} | {:>5} | {:>11.4e} | {} | {:>11.4e} | {} | {:>11.3e} | {:>8.3}",
            row.k,
            row.poly_order,
            row.total_dofs,
            row.steps,
            row.l2_rel,
            ord2,
            row.l_inf,
            ord_inf,
            row.mass_rel_err,
            row.elapsed_secs
        );
    }
    out
}

/// Formats temporal convergence rows as an aligned ASCII table showing `dt` and empirical orders.
pub fn format_temporal_convergence_table(rows: &[ConvergenceRow]) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:>3} | {:>3} | {:>10} | {:>5} | {:>11} | {:>6} | {:>11} | {:>6} | {:>11} | {:>8}",
        "K", "p", "dt", "Steps", "L2 Rel", "Order", "Linf", "Order", "Mass Err", "Time(s)"
    );
    let _ = writeln!(out, "{}", "-".repeat(103));

    for row in rows {
        let ord2 = row
            .order_l2
            .map(|v| format!("{v:>6.2}"))
            .unwrap_or_else(|| "     -".to_string());
        let ord_inf = row
            .order_linf
            .map(|v| format!("{v:>6.2}"))
            .unwrap_or_else(|| "     -".to_string());
        let _ = writeln!(
            out,
            "{:>3} | {:>3} | {:>10.4e} | {:>5} | {:>11.4e} | {} | {:>11.4e} | {} | {:>11.3e} | {:>8.3}",
            row.k,
            row.poly_order,
            row.dt,
            row.steps,
            row.l2_rel,
            ord2,
            row.l_inf,
            ord_inf,
            row.mass_rel_err,
            row.elapsed_secs
        );
    }
    out
}

/// Writes convergence rows to a CSV file.
pub fn write_convergence_csv(path: impl AsRef<Path>, rows: &[ConvergenceRow]) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut file = File::create(path)?;
    writeln!(
        file,
        "k,poly_order,total_dofs,dt,steps,elapsed_secs,l1_rel,l2_abs,l2_rel,order_l2,l_inf,order_linf,mass_rel_err"
    )?;
    for r in rows {
        writeln!(
            file,
            "{},{},{},{:.12e},{},{:.6},{:.12e},{:.12e},{:.12e},{},{:.12e},{},{:.12e}",
            r.k,
            r.poly_order,
            r.total_dofs,
            r.dt,
            r.steps,
            r.elapsed_secs,
            r.l1_rel,
            r.l2_abs,
            r.l2_rel,
            r.order_l2.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.l_inf,
            r.order_linf.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.mass_rel_err
        )?;
    }
    Ok(())
}

/// Serializes any diagnostic structure as pretty JSON.
pub fn write_json<S: Serialize>(path: impl AsRef<Path>, value: &S) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(k: usize, p: usize, dt: f64, err: f64) -> ConvergenceRow {
        ConvergenceRow {
            k,
            poly_order: p,
            total_dofs: 6 * k * k * (p + 1) * (p + 1),
            dt,
            steps: 100,
            elapsed_secs: 0.25,
            l1_rel: err,
            l2_abs: err,
            l2_rel: err,
            order_l2: None,
            l_inf: err * 2.0,
            order_linf: None,
            mass_rel_err: 1e-15,
        }
    }

    #[test]
    fn test_convergence_orders_table_csv_and_json() {
        let mut rows = vec![
            sample_row(2, 4, 1e-3, 1.6e-1),
            sample_row(4, 4, 5e-4, 1.0e-2), // 16x reduction => order = 4.0
        ];
        populate_convergence_orders(&mut rows, true);
        assert!(rows[0].order_l2.is_none());
        assert!((rows[1].order_l2.unwrap() - 4.0).abs() < 1e-12);
        assert!((rows[1].order_linf.unwrap() - 4.0).abs() < 1e-12);

        let table = format_convergence_table(&rows);
        assert!(table.contains("L2 Rel"));
        assert!(table.contains("4.00"));

        let mut t_rows = vec![
            sample_row(4, 10, 2e-3, 1.6e-7),
            sample_row(4, 10, 1e-3, 1.0e-8),
        ];
        populate_temporal_convergence_orders(&mut t_rows);
        assert!((t_rows[1].order_l2.unwrap() - 4.0).abs() < 1e-12);
        let t_table = format_temporal_convergence_table(&t_rows);
        assert!(t_table.contains("dt"));
        assert!(t_table.contains("2.0000e-3"));

        let dir = std::env::temp_dir().join(format!("cs_io_report_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let csv_path = dir.join("conv.csv");
        let json_path = dir.join("conv.json");

        write_convergence_csv(&csv_path, &rows).unwrap();
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(csv.starts_with("k,poly_order,total_dofs"));
        assert!(csv.contains("4.000000"));

        write_json(&json_path, &rows).unwrap();
        let json = fs::read_to_string(&json_path).unwrap();
        let decoded: Vec<ConvergenceRow> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].k, 4);

        let _ = fs::remove_dir_all(&dir);
    }
}
