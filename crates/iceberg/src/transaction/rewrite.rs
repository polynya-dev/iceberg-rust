// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataContentType, DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// `RewriteFilesAction` produces a snapshot with `operation = Replace`:
/// drops a set of existing data/delete files and adds a set of new files
/// in one atomic commit. Used by compaction to swap N small files for M
/// larger files without exposing intermediate state to readers.
///
/// Removed files are identified by file path (exact match). Manifests from
/// the prior snapshot are walked:
/// - Manifests with no removed files carry forward unchanged.
/// - Manifests where every entry is removed are dropped from the new
///   manifest list entirely.
/// - Manifests with partial removal are rewritten: surviving entries land
///   in a new manifest with `ManifestStatus::Existing` (preserving each
///   entry's original `snapshot_id` and `sequence_number`).
///
/// Added files go into a new `ManifestStatus::Added` manifest, routed by
/// `content_type()` (Data vs. EqualityDeletes/PositionDeletes) into
/// separate manifests as required by the spec.
pub struct RewriteFilesAction {
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
    removed_paths: HashSet<String>,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
            removed_paths: HashSet::default(),
        }
    }

    /// Add files produced by the rewrite (e.g. compacted data files).
    /// Routed by `content_type()` into separate manifests.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        for file in data_files {
            match file.content_type() {
                DataContentType::Data => self.added_data_files.push(file),
                DataContentType::EqualityDeletes | DataContentType::PositionDeletes => {
                    self.added_delete_files.push(file)
                }
            }
        }
        self
    }

    /// Mark the given files as removed in the new snapshot. Matching is by
    /// `file_path()`.
    pub fn remove_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        for file in files {
            self.removed_paths.insert(file.file_path().to_string());
        }
        self
    }

    /// Mark the given paths as removed. Convenience for callers that
    /// already track removals as `String`.
    pub fn remove_paths(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.removed_paths.extend(paths);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.added_delete_files.clone(),
        );

        snapshot_producer.validate_added_files(&self.added_data_files)?;
        snapshot_producer.validate_added_files(&self.added_delete_files)?;

        snapshot_producer
            .commit(
                RewriteFilesOperation {
                    removed_paths: self.removed_paths.clone(),
                },
                DefaultManifestProcess,
            )
            .await
    }
}

struct RewriteFilesOperation {
    removed_paths: HashSet<String>,
}

impl SnapshotProduceOperation for RewriteFilesOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        // We track removals at manifest-rewrite time inside `existing_manifest`
        // rather than emitting separate `Deleted` entries — the new manifest
        // list simply doesn't reference the dropped files. Java/PyIceberg do
        // emit `Deleted` entries for audit-trail purposes; we may add that
        // later if expiration tooling needs them.
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        // Walk each prior manifest, decide carry-forward vs. partial-rewrite
        // vs. drop.
        //
        // Pass 1: classify each manifest based on its loaded entries. We
        // can't write rewritten manifests yet because that needs `&mut
        // snapshot_produce`, and we also need the producer to compute
        // file-IO paths during this loop.
        let manifest_list_entries = manifest_list.entries().to_vec();
        let file_io = snapshot_produce.table.file_io().clone();

        enum Plan {
            CarryForward(ManifestFile),
            DropEntirely,
            Rewrite {
                survivors: Vec<ManifestEntry>,
                content: crate::spec::ManifestContentType,
            },
        }
        let mut plans: Vec<Plan> = Vec::with_capacity(manifest_list_entries.len());
        for mfile in &manifest_list_entries {
            let manifest = mfile.load_manifest(&file_io).await?;
            let total = manifest.entries().len();
            let survivors: Vec<ManifestEntry> = manifest
                .entries()
                .iter()
                .filter(|me| !self.removed_paths.contains(me.data_file().file_path()))
                .map(|me| ManifestEntry::clone(me))
                .collect();

            if survivors.len() == total {
                plans.push(Plan::CarryForward(mfile.clone()));
            } else if survivors.is_empty() {
                plans.push(Plan::DropEntirely);
            } else {
                plans.push(Plan::Rewrite {
                    survivors,
                    content: mfile.content,
                });
            }
        }

        // Pass 2: materialize. Carry-forwards and rewrites land in the
        // output; drops produce nothing.
        let mut out: Vec<ManifestFile> = Vec::with_capacity(plans.len());
        for plan in plans {
            match plan {
                Plan::CarryForward(mf) => out.push(mf),
                Plan::DropEntirely => {}
                Plan::Rewrite { survivors, content } => {
                    let rewritten = snapshot_produce
                        .write_existing_manifest_for(survivors, content)
                        .await?;
                    out.push(rewritten);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::CatalogBuilder;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, ManifestContentType, NestedField,
        Operation, PrimitiveType, Schema, Struct, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};

    async fn make_memory_table() -> (crate::memory::MemoryCatalog, crate::table::Table) {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids([1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "qty", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();

        let memory: crate::memory::MemoryCatalog = crate::memory::MemoryCatalogBuilder::default()
            .load(
                "test",
                HashMap::from([(
                    crate::memory::MEMORY_CATALOG_WAREHOUSE.to_string(),
                    "memory:///warehouse".to_string(),
                )]),
            )
            .await
            .unwrap();
        use crate::Catalog;
        let ns = crate::NamespaceIdent::from_strs(["public"]).unwrap();
        memory.create_namespace(&ns, HashMap::new()).await.unwrap();
        let creation = crate::TableCreation::builder()
            .name("orders".to_string())
            .schema(schema)
            .build();
        let table = memory.create_table(&ns, creation).await.unwrap();
        (memory, table)
    }

    fn data_file(path: &str, record_count: u64, byte_size: u64) -> crate::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(byte_size)
            .record_count(record_count)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// Snapshot 1 appends three files; snapshot 2 (rewrite) drops them all
    /// and adds one compacted file. The new manifest list must reference
    /// only the compacted file's manifest — the original manifest is
    /// dropped because every entry was removed.
    #[tokio::test]
    async fn rewrite_drops_all_files_in_manifest_and_swaps_in_new_one() {
        let (catalog, table) = make_memory_table().await;

        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let f3 = data_file("memory:///warehouse/public/orders/data/3.parquet", 30, 3072);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone(), f2.clone(), f3.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let compacted = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            60,
            5120,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted.clone()])
            .remove_data_files(vec![f1.clone(), f2.clone(), f3.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        assert_eq!(snap.summary().operation, Operation::Replace);

        let mlist = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let data_manifests: Vec<_> = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .collect();
        assert_eq!(data_manifests.len(), 1);

        let manifest = data_manifests[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(manifest.entries().len(), 1);
        assert_eq!(
            manifest.entries()[0].data_file().file_path(),
            compacted.file_path()
        );
    }

    /// Partial-manifest removal: snapshot 1 appends three files; snapshot 2
    /// removes one and adds one. The new manifest list should reference
    /// (a) a rewritten manifest carrying the two survivors as `Existing`,
    /// (b) a new manifest with the added file as `Added`.
    #[tokio::test]
    async fn rewrite_partial_manifest_keeps_survivors_as_existing() {
        let (catalog, table) = make_memory_table().await;

        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let f3 = data_file("memory:///warehouse/public/orders/data/3.parquet", 30, 3072);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone(), f2.clone(), f3.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Drop only f1, replace with a "smaller" compacted file. f2 and f3
        // survive in the rewritten manifest.
        let new_file = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            10,
            1024,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![new_file.clone()])
            .remove_data_files(vec![f1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        assert_eq!(snap.summary().operation, Operation::Replace);

        let mlist = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let data_manifests: Vec<_> = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .collect();
        // Two data manifests: rewritten survivor manifest + new added manifest.
        assert_eq!(data_manifests.len(), 2);

        let mut survivor_paths: Vec<String> = Vec::new();
        let mut added_paths: Vec<String> = Vec::new();
        for mf in &data_manifests {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for entry in m.entries() {
                match entry.status() {
                    crate::spec::ManifestStatus::Existing => {
                        survivor_paths.push(entry.data_file().file_path().to_string());
                    }
                    crate::spec::ManifestStatus::Added => {
                        added_paths.push(entry.data_file().file_path().to_string());
                    }
                    crate::spec::ManifestStatus::Deleted => {
                        panic!("unexpected Deleted status in new manifest list");
                    }
                }
            }
        }
        survivor_paths.sort();
        let mut want_survivors = vec![f2.file_path().to_string(), f3.file_path().to_string()];
        want_survivors.sort();
        assert_eq!(survivor_paths, want_survivors);
        assert_eq!(added_paths, vec![new_file.file_path().to_string()]);
    }

    /// Mixed rewrite: prior snapshot has a data manifest + a delete manifest;
    /// rewrite drops one data file and keeps the delete manifest untouched.
    /// The new manifest list should still have a `Deletes` manifest carrying
    /// forward, plus a rewritten data manifest with survivors.
    #[tokio::test]
    async fn rewrite_carries_forward_unaffected_delete_manifest() {
        let (catalog, table) = make_memory_table().await;

        let d1 = data_file("memory:///warehouse/public/orders/data/d1.parquet", 10, 1024);
        let d2 = data_file("memory:///warehouse/public/orders/data/d2.parquet", 20, 2048);
        let eq_delete = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path("memory:///warehouse/public/orders/deletes/eq.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(64)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();

        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![d1.clone(), d2.clone(), eq_delete.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Rewrite: drop d1, keep d2 + eq_delete. Add a compacted file.
        let new_file = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            10,
            1024,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![new_file.clone()])
            .remove_data_files(vec![d1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        let mlist = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        // Counts: 2 data manifests (rewritten survivors + new added) and
        // 1 delete manifest (unchanged carry-forward).
        let data_count = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .count();
        let delete_count = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Deletes)
            .count();
        assert_eq!(data_count, 2);
        assert_eq!(delete_count, 1);

        // Every survivor and added file is reachable.
        let mut all_paths: Vec<String> = Vec::new();
        for mf in mlist.entries() {
            let m = mf.load_manifest(table.file_io()).await.unwrap();
            for entry in m.entries() {
                all_paths.push(entry.data_file().file_path().to_string());
            }
        }
        all_paths.sort();
        let mut want = vec![
            d2.file_path().to_string(),
            new_file.file_path().to_string(),
            eq_delete.file_path().to_string(),
        ];
        want.sort();
        assert_eq!(all_paths, want);
    }

    /// Rewrite with no `removed_paths` is an append disguised as a replace —
    /// the prior manifest is carried forward and the new file joins it.
    /// The snapshot is still tagged `Replace`.
    #[tokio::test]
    async fn rewrite_with_no_removals_carries_prior_manifest() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let tx = Transaction::new(&table);
        let action = tx.rewrite_files().add_data_files(vec![f2.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        assert_eq!(snap.summary().operation, Operation::Replace);

        let mlist = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        let data_manifest_count = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .count();
        assert_eq!(data_manifest_count, 2);
    }

    /// Rewrite with empty add + empty remove is a no-op-ish state — the
    /// SnapshotProducer requires *something* added (upstream contract).
    /// Documents the existing behavior so callers know to skip the action
    /// when there's nothing to compact.
    #[tokio::test]
    async fn rewrite_with_empty_input_errors() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let tx = Transaction::new(&table);
        let action = tx.rewrite_files();
        let tx = action.apply(tx).unwrap();
        assert!(tx.commit(&catalog).await.is_err());
    }
}
