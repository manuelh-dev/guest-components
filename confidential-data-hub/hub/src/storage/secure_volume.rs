// Copyright (c) 2026 NVIDIA Corporation
//
// SPDX-License-Identifier: Apache-2.0

//! Manifest-driven secure block-volume activation.

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    os::fd::AsRawFd,
    os::unix::fs::FileTypeExt,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context};
use nix::libc;
use resource_uri::{ResourceUri, DEFAULT_RESOURCE_PLUGIN};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio::sync::Mutex;
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::storage::{
    drivers::{luks2::Luks2Formatter, run_command},
    volume_type::blockdevice::{get_device_path, parse_device_id},
};

const MAX_MANIFEST_SIZE: usize = 64 * 1024;
const SUPPORTED_SCHEMA_VERSION: u32 = 1;
const PROTECTION_TYPE_LUKS2_VERITY_RO: &str = "luks2-verity-ro";
const PROTECTION_TYPE_LUKS2_INTEGRITY_RW: &str = "luks2-integrity-rw";
const SUPPORTED_HASH_ALGORITHM: &str = "sha256";
const SUPPORTED_BLOCK_SIZE: u64 = 4096;
const MAX_LUKS_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const SHA256_DIGEST_BYTES: u64 = 32;

// Linux block-device ioctl request numbers from <linux/fs.h>.
const BLKGETSIZE64: libc::Ioctl = 0x8008_1272u32 as libc::Ioctl;
const BLKROGET: libc::Ioctl = 0x125eu32 as libc::Ioctl;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("secure-volume manifest exceeds {MAX_MANIFEST_SIZE} bytes")]
    ManifestTooLarge,

    #[error("failed to parse secure-volume manifest: {0}")]
    ManifestParse(#[from] serde_json::Error),

    #[error("invalid secure-volume manifest: {0}")]
    InvalidManifest(String),

    #[error("invalid secure-volume device: {0}")]
    InvalidDevice(String),

    #[error("secure-volume activation failed: {0:#}")]
    Activation(#[source] anyhow::Error),

    #[error("unknown secure-volume activation {0}")]
    UnknownActivation(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Manifest {
    pub schema_version: u32,
    pub volume_id: String,
    pub volume_version: String,
    pub protection: Protection,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Protection {
    #[serde(rename = "type")]
    pub protection_type: String,
    pub key_uri: String,
    pub luks_uuid: String,
    #[serde(default)]
    pub verity: Option<Verity>,
    #[serde(default)]
    pub header: Option<Header>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Verity {
    pub algorithm: String,
    pub root_hash: String,
    pub salt: String,
    pub data_block_size: u64,
    pub hash_block_size: u64,
    pub data_blocks: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct Header {
    pub algorithm: String,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub sha256: String,
}

impl Manifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_MANIFEST_SIZE {
            return Err(Error::ManifestTooLarge);
        }

        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(Error::InvalidManifest(format!(
                "unsupported schemaVersion {}",
                self.schema_version
            )));
        }
        validate_identifier("volumeId", &self.volume_id)?;
        validate_identifier("volumeVersion", &self.volume_version)?;

        if !matches!(
            self.protection.protection_type.as_str(),
            PROTECTION_TYPE_LUKS2_VERITY_RO | PROTECTION_TYPE_LUKS2_INTEGRITY_RW
        ) {
            return Err(Error::InvalidManifest(format!(
                "unsupported protection type {}",
                self.protection.protection_type
            )));
        }
        validate_kbs_uri(&self.protection.key_uri)?;
        Uuid::parse_str(&self.protection.luks_uuid)
            .map_err(|e| Error::InvalidManifest(format!("invalid LUKS UUID: {e}")))?;

        match self.protection.protection_type.as_str() {
            PROTECTION_TYPE_LUKS2_VERITY_RO => self.validate_verity_ro()?,
            PROTECTION_TYPE_LUKS2_INTEGRITY_RW => self.validate_integrity_rw()?,
            _ => unreachable!("unsupported protection type already rejected"),
        }
        Ok(())
    }

    fn validate_verity_ro(&self) -> Result<()> {
        if self.protection.header.is_some() {
            return Err(Error::InvalidManifest(
                "header must be omitted for luks2-verity-ro".to_string(),
            ));
        }
        let verity = self.protection.verity.as_ref().ok_or_else(|| {
            Error::InvalidManifest("verity is required for luks2-verity-ro".to_string())
        })?;
        if verity.algorithm != SUPPORTED_HASH_ALGORITHM {
            return Err(Error::InvalidManifest(format!(
                "unsupported verity algorithm {}",
                verity.algorithm
            )));
        }
        if verity.data_block_size != SUPPORTED_BLOCK_SIZE
            || verity.hash_block_size != SUPPORTED_BLOCK_SIZE
        {
            return Err(Error::InvalidManifest(
                "dataBlockSize and hashBlockSize must both be 4096".to_string(),
            ));
        }
        if verity.data_blocks == 0 {
            return Err(Error::InvalidManifest(
                "dataBlocks must be greater than zero".to_string(),
            ));
        }
        validate_hex("rootHash", &verity.root_hash, Some(64))?;
        validate_hex("salt", &verity.salt, None)?;
        self.layout()?;
        Ok(())
    }

    fn validate_integrity_rw(&self) -> Result<()> {
        if self.protection.verity.is_some() {
            return Err(Error::InvalidManifest(
                "verity must be omitted for luks2-integrity-rw".to_string(),
            ));
        }
        let header = self.protection.header.as_ref().ok_or_else(|| {
            Error::InvalidManifest("header is required for luks2-integrity-rw".to_string())
        })?;
        if header.algorithm != SUPPORTED_HASH_ALGORITHM {
            return Err(Error::InvalidManifest(format!(
                "unsupported header algorithm {}",
                header.algorithm
            )));
        }
        if header.length_bytes == 0 || header.length_bytes > MAX_LUKS_HEADER_BYTES {
            return Err(Error::InvalidManifest(format!(
                "header lengthBytes must be between 1 and {MAX_LUKS_HEADER_BYTES}"
            )));
        }
        header
            .offset_bytes
            .checked_add(header.length_bytes)
            .ok_or_else(|| Error::InvalidManifest("header range overflows u64".to_string()))?;
        validate_hex("header.sha256", &header.sha256, Some(64))?;
        Ok(())
    }

    fn layout(&self) -> Result<Layout> {
        let v = self.protection.verity.as_ref().ok_or_else(|| {
            Error::InvalidManifest("verity is required for layout calculation".to_string())
        })?;
        let data_bytes = v
            .data_blocks
            .checked_mul(v.data_block_size)
            .ok_or_else(|| Error::InvalidManifest("data size overflows u64".to_string()))?;

        let hashes_per_block = v.hash_block_size / SHA256_DIGEST_BYTES;
        let mut level_input_blocks = v.data_blocks;
        let mut hash_blocks = 0u64;
        loop {
            let level_blocks = level_input_blocks
                .checked_add(hashes_per_block - 1)
                .ok_or_else(|| {
                    Error::InvalidManifest("hash-tree size overflows u64".to_string())
                })?
                / hashes_per_block;
            hash_blocks = hash_blocks.checked_add(level_blocks).ok_or_else(|| {
                Error::InvalidManifest("hash-tree size overflows u64".to_string())
            })?;
            if level_blocks == 1 {
                break;
            }
            level_input_blocks = level_blocks;
        }

        let hash_bytes = hash_blocks.checked_mul(v.hash_block_size).ok_or_else(|| {
            Error::InvalidManifest("hash-tree byte size overflows u64".to_string())
        })?;
        let minimum_device_bytes = data_bytes
            .checked_add(hash_bytes)
            .ok_or_else(|| Error::InvalidManifest("device size overflows u64".to_string()))?;

        Ok(Layout {
            data_bytes,
            minimum_device_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    data_bytes: u64,
    minimum_device_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Activation {
    pub activation_id: String,
    pub device_path: String,
}

#[derive(Clone, Debug)]
struct ActivationState {
    luks_name: String,
    verity_name: Option<String>,
    header_path: Option<PathBuf>,
}

struct PendingActivation {
    device_path: String,
    state: ActivationState,
}

#[derive(Default)]
pub struct Manager {
    activations: Mutex<HashMap<String, ActivationState>>,
}

impl Manager {
    pub async fn activate(
        &self,
        device_id: &str,
        manifest: &Manifest,
        key: Zeroizing<Vec<u8>>,
    ) -> Result<Activation> {
        let (major, minor) = parse_device_id(device_id)
            .map_err(|e| Error::InvalidDevice(format!("invalid device ID {device_id}: {e}")))?;
        let device_path = get_device_path(major, minor)
            .await
            .map_err(|e| Error::InvalidDevice(format!("cannot resolve {device_id}: {e}")))?;
        validate_source_device(&device_path, manifest)?;

        let activation_id = Uuid::new_v4().to_string();
        let activation = match manifest.protection.protection_type.as_str() {
            PROTECTION_TYPE_LUKS2_VERITY_RO => {
                self.activate_verity_ro(&activation_id, &device_path, manifest, key)
                    .await?
            }
            PROTECTION_TYPE_LUKS2_INTEGRITY_RW => {
                self.activate_integrity_rw(&activation_id, &device_path, manifest, key)
                    .await?
            }
            _ => unreachable!("unsupported protection type already rejected"),
        };

        self.activations
            .lock()
            .await
            .insert(activation_id.clone(), activation.state);

        Ok(Activation {
            activation_id,
            device_path: activation.device_path,
        })
    }

    async fn activate_verity_ro(
        &self,
        activation_id: &str,
        device_path: &str,
        manifest: &Manifest,
        key: Zeroizing<Vec<u8>>,
    ) -> Result<PendingActivation> {
        let verity_name = format!("cdh-verity-{activation_id}");
        let luks_name = format!("cdh-luks-{activation_id}");
        let verity_path = format!("/dev/mapper/{verity_name}");
        let luks_path = format!("/dev/mapper/{luks_name}");

        open_verity(device_path, &verity_name, manifest).map_err(Error::Activation)?;

        let activation_result = (|| -> anyhow::Result<()> {
            validate_luks_uuid(&verity_path, None, &manifest.protection.luks_uuid)?;
            Luks2Formatter::default().open_device_read_only(&verity_path, &luks_name, key)?;
            if !block_device_is_read_only(&luks_path)? {
                bail!("activated LUKS mapper is not read-only");
            }
            Ok(())
        })();

        if let Err(e) = activation_result {
            let _ = close_luks(&luks_name);
            let _ = close_verity(&verity_name);
            return Err(Error::Activation(e));
        }

        Ok(PendingActivation {
            device_path: luks_path,
            state: ActivationState {
                luks_name,
                verity_name: Some(verity_name),
                header_path: None,
            },
        })
    }

    async fn activate_integrity_rw(
        &self,
        activation_id: &str,
        device_path: &str,
        manifest: &Manifest,
        key: Zeroizing<Vec<u8>>,
    ) -> Result<PendingActivation> {
        let luks_name = format!("cdh-luks-{activation_id}");
        let luks_path = format!("/dev/mapper/{luks_name}");
        let header_file =
            copy_and_authenticate_header(device_path, manifest).map_err(Error::Activation)?;
        let header_path = header_file.into_temp_path().keep().map_err(|e| {
            Error::Activation(anyhow!(e).context("persist authenticated LUKS header in tmpfs"))
        })?;

        let activation_result = (|| -> anyhow::Result<()> {
            validate_luks_uuid(
                device_path,
                Some(header_path.as_path()),
                &manifest.protection.luks_uuid,
            )?;
            Luks2Formatter::default().open_device(
                device_path,
                Some(path_to_str(&header_path)?),
                &luks_name,
                key,
            )?;
            if block_device_is_read_only(&luks_path)? {
                bail!("activated LUKS mapper is read-only");
            }
            Ok(())
        })();

        if let Err(e) = activation_result {
            let _ = close_luks(&luks_name);
            let _ = std::fs::remove_file(&header_path);
            return Err(Error::Activation(e));
        }

        Ok(PendingActivation {
            device_path: luks_path,
            state: ActivationState {
                luks_name,
                verity_name: None,
                header_path: Some(header_path),
            },
        })
    }

    pub async fn deactivate(&self, activation_id: &str) -> Result<()> {
        let state = self
            .activations
            .lock()
            .await
            .get(activation_id)
            .cloned()
            .ok_or_else(|| Error::UnknownActivation(activation_id.to_string()))?;

        close_luks(&state.luks_name).map_err(Error::Activation)?;
        if let Some(verity_name) = &state.verity_name {
            if let Err(e) = close_verity(verity_name) {
                return Err(Error::Activation(e));
            }
        }
        if let Some(header_path) = &state.header_path {
            std::fs::remove_file(header_path).map_err(|e| {
                Error::Activation(anyhow!(e).context("remove authenticated LUKS header"))
            })?;
        }
        self.activations.lock().await.remove(activation_id);
        Ok(())
    }
}

pub fn validate_kbs_uri(uri: &str) -> Result<()> {
    if !uri.starts_with("kbs://") {
        return Err(Error::InvalidManifest(
            "resource URI must use the kbs scheme".to_string(),
        ));
    }
    let parsed = ResourceUri::try_from(uri)
        .map_err(|e| Error::InvalidManifest(format!("invalid KBS resource URI: {e}")))?;
    if parsed.plugin() != DEFAULT_RESOURCE_PLUGIN || parsed.whole_uri() != uri {
        return Err(Error::InvalidManifest(
            "resource URI must be a canonical kbs resource URI".to_string(),
        ));
    }
    Ok(())
}

fn validate_identifier(field: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 128 {
        return Err(Error::InvalidManifest(format!(
            "{field} must contain between 1 and 128 characters"
        )));
    }
    if !value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return Err(Error::InvalidManifest(format!(
            "{field} contains unsupported characters"
        )));
    }
    Ok(())
}

fn validate_hex(field: &str, value: &str, exact_len: Option<usize>) -> Result<()> {
    if value.is_empty()
        || !value.len().is_multiple_of(2)
        || !value.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(Error::InvalidManifest(format!(
            "{field} must be a non-empty even-length hexadecimal string"
        )));
    }
    if exact_len.is_some_and(|len| value.len() != len) {
        return Err(Error::InvalidManifest(format!(
            "{field} has an invalid length"
        )));
    }
    // Decode as a final canonical validation step and bound attacker-controlled allocation.
    if value.len() > 1024 {
        return Err(Error::InvalidManifest(format!("{field} is too long")));
    }
    hex::decode(value).map_err(|e| Error::InvalidManifest(format!("invalid {field}: {e}")))?;
    Ok(())
}

fn validate_source_device(path: &str, manifest: &Manifest) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| Error::InvalidDevice(format!("cannot stat {path}: {e}")))?;
    if !metadata.file_type().is_block_device() {
        return Err(Error::InvalidDevice(format!(
            "{path} is not a block device"
        )));
    }
    let read_only = block_device_is_read_only(path).map_err(Error::Activation)?;
    match manifest.protection.protection_type.as_str() {
        PROTECTION_TYPE_LUKS2_VERITY_RO if !read_only => {
            return Err(Error::InvalidDevice(format!("{path} is not read-only")));
        }
        PROTECTION_TYPE_LUKS2_INTEGRITY_RW if read_only => {
            return Err(Error::InvalidDevice(format!("{path} is read-only")));
        }
        _ => {}
    }

    let actual_size = block_device_size(path).map_err(Error::Activation)?;
    match manifest.protection.protection_type.as_str() {
        PROTECTION_TYPE_LUKS2_VERITY_RO => {
            let layout = manifest.layout()?;
            if actual_size < layout.minimum_device_bytes {
                return Err(Error::InvalidDevice(format!(
                    "{path} is too small: {actual_size} bytes, need at least {}",
                    layout.minimum_device_bytes
                )));
            }
        }
        PROTECTION_TYPE_LUKS2_INTEGRITY_RW => {
            let header = manifest.protection.header.as_ref().ok_or_else(|| {
                Error::InvalidManifest("header is required for luks2-integrity-rw".to_string())
            })?;
            let header_end = header
                .offset_bytes
                .checked_add(header.length_bytes)
                .ok_or_else(|| Error::InvalidManifest("header range overflows u64".to_string()))?;
            if actual_size < header_end {
                return Err(Error::InvalidDevice(format!(
                    "{path} is too small: {actual_size} bytes, header ends at {header_end}"
                )));
            }
        }
        _ => unreachable!("unsupported protection type already rejected"),
    }
    Ok(())
}

fn open_verity(device_path: &str, name: &str, manifest: &Manifest) -> anyhow::Result<()> {
    let v = manifest
        .protection
        .verity
        .as_ref()
        .context("verity is required for dm-verity activation")?;
    let layout = manifest.layout().map_err(|e| anyhow!(e))?;
    let hash_offset = layout.data_bytes.to_string();
    let data_block_size = v.data_block_size.to_string();
    let hash_block_size = v.hash_block_size.to_string();
    let data_blocks = v.data_blocks.to_string();
    let args = [
        "open",
        device_path,
        name,
        device_path,
        v.root_hash.as_str(),
        "--no-superblock",
        "--hash",
        v.algorithm.as_str(),
        "--salt",
        v.salt.as_str(),
        "--data-block-size",
        data_block_size.as_str(),
        "--hash-block-size",
        hash_block_size.as_str(),
        "--data-blocks",
        data_blocks.as_str(),
        "--hash-offset",
        hash_offset.as_str(),
    ];
    run_command("veritysetup", &args, None).context("open dm-verity mapping")?;
    Ok(())
}

fn validate_luks_uuid(
    device_path: &str,
    header_path: Option<&Path>,
    expected: &str,
) -> anyhow::Result<()> {
    let mut args = vec!["luksUUID"];
    let header_string;
    if let Some(header_path) = header_path {
        header_string = path_to_str(header_path)?.to_string();
        args.push("--header");
        args.push(header_string.as_str());
    }
    args.push(device_path);
    let (stdout, _) =
        run_command("cryptsetup", &args, None).context("read authenticated LUKS UUID")?;
    if stdout.trim() != expected {
        bail!(
            "authenticated LUKS UUID mismatch: expected {expected}, got {}",
            stdout.trim()
        );
    }
    Ok(())
}

fn copy_and_authenticate_header(
    device_path: &str,
    manifest: &Manifest,
) -> anyhow::Result<NamedTempFile> {
    let header = manifest
        .protection
        .header
        .as_ref()
        .context("header is required for luks2-integrity-rw")?;
    let mut source = File::open(device_path).with_context(|| format!("open {device_path}"))?;
    source
        .seek(SeekFrom::Start(header.offset_bytes))
        .with_context(|| format!("seek to header offset {}", header.offset_bytes))?;

    let mut header_file = tempfile::Builder::new()
        .prefix("cdh-luks-header-")
        .tempfile_in("/run")
        .context("create tmpfs LUKS header copy")?;
    let mut hasher = Sha256::new();
    let mut remaining = header.length_bytes;
    let mut buf = [0u8; 64 * 1024];
    while remaining > 0 {
        let to_read = std::cmp::min(remaining, buf.len() as u64) as usize;
        source
            .read_exact(&mut buf[..to_read])
            .context("read LUKS header from source device")?;
        hasher.update(&buf[..to_read]);
        header_file
            .write_all(&buf[..to_read])
            .context("write LUKS header to tmpfs")?;
        remaining -= to_read as u64;
    }
    header_file.flush().context("flush LUKS header copy")?;

    let actual = hex::encode(hasher.finalize());
    if actual != header.sha256.to_ascii_lowercase() {
        bail!(
            "authenticated LUKS header digest mismatch: expected {}, got {actual}",
            header.sha256
        );
    }
    Ok(header_file)
}

fn path_to_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))
}

fn close_luks(name: &str) -> anyhow::Result<()> {
    if Path::new(&format!("/dev/mapper/{name}")).exists() {
        Luks2Formatter::default().close_device(name)?;
    }
    Ok(())
}

fn close_verity(name: &str) -> anyhow::Result<()> {
    if Path::new(&format!("/dev/mapper/{name}")).exists() {
        run_command("veritysetup", &["close", name], None).context("close dm-verity mapping")?;
    }
    Ok(())
}

fn block_device_size(path: &str) -> anyhow::Result<u64> {
    let file = File::open(path).with_context(|| format!("open block device {path}"))?;
    let mut size = 0u64;
    // SAFETY: BLKGETSIZE64 writes one u64 to the valid pointer supplied here.
    let rc = unsafe { libc::ioctl(file.as_raw_fd(), BLKGETSIZE64, &mut size) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("query size of block device {path}"));
    }
    Ok(size)
}

fn block_device_is_read_only(path: &str) -> anyhow::Result<bool> {
    let file = File::open(path).with_context(|| format!("open block device {path}"))?;
    let mut read_only: libc::c_int = 0;
    // SAFETY: BLKROGET writes one int to the valid pointer supplied here.
    let rc = unsafe { libc::ioctl(file.as_raw_fd(), BLKROGET, &mut read_only) };
    if rc < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("query read-only state of block device {path}"));
    }
    Ok(read_only != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> Vec<u8> {
        br#"{
          "schemaVersion": 1,
          "volumeId": "model-data",
          "volumeVersion": "v1",
          "protection": {
            "type": "luks2-verity-ro",
            "keyUri": "kbs:///kata-ci/storage-key/model-data-v1",
            "luksUuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "verity": {
              "algorithm": "sha256",
              "rootHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "salt": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "dataBlockSize": 4096,
              "hashBlockSize": 4096,
              "dataBlocks": 128000
            }
          }
        }"#
        .to_vec()
    }

    fn valid_rw_manifest() -> Vec<u8> {
        br#"{
          "schemaVersion": 1,
          "volumeId": "model-data",
          "volumeVersion": "v1",
          "protection": {
            "type": "luks2-integrity-rw",
            "keyUri": "kbs:///kata-ci/storage-key/model-data-v1",
            "luksUuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "header": {
              "algorithm": "sha256",
              "offsetBytes": 0,
              "lengthBytes": 16777216,
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }
          }
        }"#
        .to_vec()
    }

    #[test]
    fn parses_and_derives_contiguous_layout() {
        let manifest = Manifest::parse(&valid_manifest()).unwrap();
        let layout = manifest.layout().unwrap();
        assert_eq!(layout.data_bytes, 128000 * 4096);
        assert!(layout.minimum_device_bytes > layout.data_bytes);
    }

    #[test]
    fn parses_integrity_rw_manifest() {
        let manifest = Manifest::parse(&valid_rw_manifest()).unwrap();
        assert_eq!(
            manifest.protection.protection_type,
            PROTECTION_TYPE_LUKS2_INTEGRITY_RW
        );
        assert!(manifest.protection.verity.is_none());
        assert_eq!(
            manifest.protection.header.as_ref().unwrap().length_bytes,
            16 * 1024 * 1024
        );
    }

    #[test]
    fn rejects_integrity_rw_manifest_without_header() {
        let bytes = String::from_utf8(valid_rw_manifest()).unwrap().replace(
            r#",
            "header": {
              "algorithm": "sha256",
              "offsetBytes": 0,
              "lengthBytes": 16777216,
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }"#,
            "",
        );
        assert!(matches!(
            Manifest::parse(bytes.as_bytes()),
            Err(Error::InvalidManifest(_))
        ));
    }

    #[test]
    fn rejects_unknown_fields() {
        let bytes = String::from_utf8(valid_manifest()).unwrap().replace(
            "\"volumeId\": \"model-data\",",
            "\"volumeId\": \"model-data\", \"devicePath\": \"/dev/vda\",",
        );
        assert!(matches!(
            Manifest::parse(bytes.as_bytes()),
            Err(Error::ManifestParse(_))
        ));
    }

    #[test]
    fn rejects_non_kbs_key_uri() {
        let bytes = String::from_utf8(valid_manifest()).unwrap().replace(
            "kbs:///kata-ci/storage-key/model-data-v1",
            "file:///tmp/key",
        );
        assert!(matches!(
            Manifest::parse(bytes.as_bytes()),
            Err(Error::InvalidManifest(_))
        ));
    }
}
