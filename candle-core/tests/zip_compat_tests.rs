//! Exercise the archive readers affected by the zip 9 filename API change.
use candle_core::{npy::NpzTensors, pickle::PthTensors, Device, Error, Result, Tensor};
use std::path::{Path, PathBuf};

struct Archive(PathBuf);

impl Archive {
    fn new(label: &str) -> Self {
        Self(std::env::temp_dir().join(format!(
            "candle-zip-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )))
    }
}

impl AsRef<Path> for Archive {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Archive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn check_npz(path: &Archive, name: &str) -> Result<()> {
    let eager = Tensor::read_npz(path)?;
    assert_eq!(eager.len(), 1);
    assert_eq!(eager[0].0, name);
    assert_eq!(eager[0].1.to_vec1::<f32>()?, [1.5, -2., 3.25]);
    let lazy = NpzTensors::new(path)?;
    assert_eq!(lazy.names().len(), 1);
    let loaded = lazy
        .get(name)?
        .expect("filename must be preserved in the index");
    assert_eq!(loaded.to_vec1::<f32>()?, [1.5, -2., 3.25]);
    assert!(lazy.get("missing")?.is_none());
    Ok(())
}

#[test]
fn npz_ascii_and_utf8_names() -> Result<()> {
    let file = Archive::new("utf8");
    let tensor = Tensor::new(&[1.5f32, -2., 3.25], &Device::Cpu)?;
    for name in ["weights", "café/重み"] {
        Tensor::write_npz(&[(name, &tensor)], &file)?;
        check_npz(&file, name)?;
    }
    Ok(())
}

#[test]
fn npz_cp437_name() -> Result<()> {
    let file = Archive::new("cp437");
    let tensor = Tensor::new(&[1.5f32, -2., 3.25], &Device::Cpu)?;
    Tensor::write_npz(&[("x", &tensor)], &file)?;
    let mut bytes = std::fs::read(&file)?;
    // Replace the same one-byte name in both local and central headers.
    // 0x82 is é in CP437 and forces zip's owned decoded-name path.
    let offsets: Vec<_> = bytes
        .windows(5)
        .enumerate()
        .filter_map(|(i, name)| (name == b"x.npy").then_some(i))
        .collect();
    assert_eq!(offsets.len(), 2);
    for offset in offsets {
        bytes[offset] = 0x82;
    }
    std::fs::write(&file, bytes)?;
    check_npz(&file, "é")
}

#[test]
fn invalid_archives_propagate_zip_errors() -> Result<()> {
    let file = Archive::new("invalid");
    for bytes in [&b"not a zip archive"[..], &b"PK\x03\x04"[..]] {
        std::fs::write(&file, bytes)?;
        assert!(matches!(Tensor::read_npz(&file), Err(Error::Zip(_))));
        assert!(matches!(NpzTensors::new(&file), Err(Error::Zip(_))));
        assert!(matches!(PthTensors::new(&file, None), Err(Error::Zip(_))));
    }
    Ok(())
}

#[test]
fn pth_metadata_and_tensor_storage_still_load() -> Result<()> {
    for (file, key) in [
        ("tests/test.pt", None),
        ("tests/test_with_key.pt", Some("model_state_dict")),
    ] {
        let checkpoint = PthTensors::new(file, key)?;
        assert!(checkpoint.tensor_infos().contains_key("test"));
        let tensor = checkpoint.get("test")?.expect("fixture contains test");
        assert!(tensor.elem_count() > 0);
        assert!(checkpoint.get("missing")?.is_none());
    }
    Ok(())
}
