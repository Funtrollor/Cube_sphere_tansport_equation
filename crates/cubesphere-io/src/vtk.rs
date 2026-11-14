//! VTK XML (`.vtu`) and ParaView time-series (`.pvd`) writers.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use cubesphere_core::{CubedSphereMesh, FloatScalar};

/// Writes the spectral-element cubed-sphere mesh and scalar point fields to a VTK XML `.vtu` file.
///
/// Subdivides each `(p + 1) x (p + 1)` element in `[elem][j][i]` order into `p x p` `VTK_QUAD`
/// sub-cells and formats nodal `xyz` and `fields` values as `f64` ASCII for ParaView.
///
/// # Errors
///
/// Returns `io::ErrorKind::InvalidInput` if any slice in `fields` does not match `mesh.total_nodes`,
/// or an `io::Error` if creating directories, creating `path`, or writing fails.
pub fn write_vtu<T: FloatScalar>(
    path: impl AsRef<Path>,
    mesh: &CubedSphereMesh<T>,
    fields: &[(&str, &[T])],
) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    for &(name, slice) in fields {
        if slice.len() != mesh.total_nodes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "VTK field '{name}' length {} does not match mesh nodes {}",
                    slice.len(),
                    mesh.total_nodes
                ),
            ));
        }
    }

    let n = mesh.n_nodes_1d;
    let p = mesh.reference.order;
    let n_points = mesh.total_nodes;
    let n_cells = mesh.n_elems * p * p;

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "<?xml version=\"1.0\"?>")?;
    writeln!(
        w,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(w, "  <UnstructuredGrid>")?;
    writeln!(
        w,
        "    <Piece NumberOfPoints=\"{n_points}\" NumberOfCells=\"{n_cells}\">"
    )?;

    // PointData
    if !fields.is_empty() {
        writeln!(w, "      <PointData Scalars=\"{}\">", fields[0].0)?;
        for &(name, slice) in fields {
            writeln!(
                w,
                "        <DataArray type=\"Float64\" Name=\"{name}\" format=\"ascii\">"
            )?;
            for chunk in slice.chunks(6) {
                write!(w, "         ")?;
                for val in chunk {
                    write!(w, " {:.12e}", val.to_f64())?;
                }
                writeln!(w)?;
            }
            writeln!(w, "        </DataArray>")?;
        }
        writeln!(w, "      </PointData>")?;
    }

    // Points
    writeln!(w, "      <Points>")?;
    writeln!(
        w,
        "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for idx in 0..n_points {
        writeln!(
            w,
            "          {:.12e} {:.12e} {:.12e}",
            mesh.xyz[0][idx].to_f64(),
            mesh.xyz[1][idx].to_f64(),
            mesh.xyz[2][idx].to_f64()
        )?;
    }
    writeln!(w, "        </DataArray>")?;
    writeln!(w, "      </Points>")?;

    // Cells
    writeln!(w, "      <Cells>")?;
    writeln!(
        w,
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"
    )?;
    for elem in 0..mesh.n_elems {
        let base = elem * mesh.n_nodes_elem;
        for j in 0..p {
            for i in 0..p {
                let n0 = base + j * n + i;
                let n1 = base + j * n + (i + 1);
                let n2 = base + (j + 1) * n + (i + 1);
                let n3 = base + (j + 1) * n + i;
                writeln!(w, "          {n0} {n1} {n2} {n3}")?;
            }
        }
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(
        w,
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    )?;
    for c in 1..=n_cells {
        if c % 16 == 1 {
            write!(w, "         ")?;
        }
        write!(w, " {}", c * 4)?;
        if c % 16 == 0 || c == n_cells {
            writeln!(w)?;
        }
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(
        w,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    for c in 1..=n_cells {
        if c % 24 == 1 {
            write!(w, "         ")?;
        }
        write!(w, " 9")?; // VTK_QUAD
        if c % 24 == 0 || c == n_cells {
            writeln!(w)?;
        }
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(w, "      </Cells>")?;
    writeln!(w, "    </Piece>")?;
    writeln!(w, "  </UnstructuredGrid>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}

/// Writes a time-series of `.vtu` files (`{prefix}_{idx:04}.vtu`) plus a ParaView `.pvd` index
/// file (`{prefix}.pvd`) into `out_dir`, storing each `(time, state)` frame under field name `"phi"`.
///
/// # Errors
///
/// Returns `io::ErrorKind::InvalidInput` if any frame slice length differs from `mesh.total_nodes`,
/// or an `io::Error` on filesystem or write failures.
pub fn write_pvd_series<T: FloatScalar>(
    out_dir: impl AsRef<Path>,
    prefix: &str,
    mesh: &CubedSphereMesh<T>,
    frames: &[(f64, &[T])],
) -> io::Result<()> {
    let dir = out_dir.as_ref();
    fs::create_dir_all(dir)?;

    let pvd_path = dir.join(format!("{prefix}.pvd"));
    let pvd_file = File::create(pvd_path)?;
    let mut pvd = BufWriter::new(pvd_file);

    writeln!(pvd, "<?xml version=\"1.0\"?>")?;
    writeln!(
        pvd,
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(pvd, "  <Collection>")?;

    for (idx, &(time, state)) in frames.iter().enumerate() {
        let frame_name = format!("{prefix}_{idx:04}.vtu");
        let frame_path = dir.join(&frame_name);
        write_vtu(&frame_path, mesh, &[("phi", state)])?;
        writeln!(
            pvd,
            "    <DataSet timestep=\"{time:.12e}\" group=\"\" part=\"0\" file=\"{frame_name}\"/>"
        )?;
    }

    writeln!(pvd, "  </Collection>")?;
    writeln!(pvd, "</VTKFile>")?;
    pvd.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubesphere_core::{MeshConfig, ProjectionKind};

    #[test]
    fn test_write_vtu_and_pvd_series_roundtrip() {
        let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: 2,
            poly_order: 3,
            projection: ProjectionKind::Equiangular,
        });
        let phi0 = vec![1.0_f64; mesh.total_nodes];
        let phi1 = vec![2.0_f64; mesh.total_nodes];

        let dir = std::env::temp_dir().join(format!("cs_io_vtk_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let vtu_path = dir.join("single.vtu");
        write_vtu(
            &vtu_path,
            &mesh,
            &[("phi", &phi0), ("sqrt_g", &mesh.sqrt_g)],
        )
        .unwrap();
        let vtu_xml = fs::read_to_string(&vtu_path).unwrap();
        assert!(vtu_xml.contains("<VTKFile type=\"UnstructuredGrid\""));
        let expected_cells = mesh.n_elems * 3 * 3;
        assert!(vtu_xml.contains(&format!(
            "NumberOfPoints=\"{}\" NumberOfCells=\"{expected_cells}\"",
            mesh.total_nodes
        )));
        assert!(vtu_xml.contains("Name=\"phi\""));
        assert!(vtu_xml.contains("Name=\"sqrt_g\""));

        write_pvd_series(&dir, "traj", &mesh, &[(0.0, &phi0), (0.5, &phi1)]).unwrap();
        let pvd_xml = fs::read_to_string(dir.join("traj.pvd")).unwrap();
        assert!(pvd_xml.contains("<VTKFile type=\"Collection\""));
        assert!(pvd_xml.contains("file=\"traj_0000.vtu\""));
        assert!(pvd_xml.contains("file=\"traj_0001.vtu\""));
        assert!(dir.join("traj_0000.vtu").exists());
        assert!(dir.join("traj_0001.vtu").exists());

        // Mismatched field length must return InvalidInput
        let bad_err = write_vtu(&vtu_path, &mesh, &[("bad", &[1.0_f64; 3])]).unwrap_err();
        assert_eq!(bad_err.kind(), io::ErrorKind::InvalidInput);

        let _ = fs::remove_dir_all(&dir);
    }
}
