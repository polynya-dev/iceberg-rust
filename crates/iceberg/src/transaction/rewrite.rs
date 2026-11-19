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
use futures::{StreamExt, TryStreamExt};
use uuid::Uuid;

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Manifest, ManifestEntry, ManifestFile, Operation,
};
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
///
/// The commit fails if a removed file is no longer in the table it lands
/// on — another rewrite got there first, and adding this one's files too
/// would duplicate their rows. A retried commit checks again.
///
/// A deletion vector — a blob in a Puffin file, which may hold others —
/// is removed by its place ([`Self::remove_deletion_vectors`]). On a
/// table with them, removing a data file must remove its vector too: the
/// commit fails if a removed data file has one it doesn't remove, a
/// vector that landed after the rewrite read the file, deleting rows its
/// added files still hold.
pub struct RewriteFilesAction {
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
    removal: Removal,
    data_sequence_number: Option<i64>,
    validate_no_row_ids: bool,
}

impl RewriteFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
            removal: Removal::default(),
            data_sequence_number: None,
            validate_no_row_ids: false,
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
            self.removal.paths.insert(file.file_path().to_string());
        }
        self
    }

    /// Mark the given paths as removed. Convenience for callers that
    /// already track removals as `String`.
    pub fn remove_paths(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.removal.paths.extend(paths);
        self
    }

    /// Mark the given deletion vectors as removed, each by its Puffin
    /// file's path and its blob's offset there (`content_offset`).
    pub fn remove_deletion_vectors(
        mut self,
        vectors: impl IntoIterator<Item = (String, i64)>,
    ) -> Self {
        self.removal.vectors.extend(vectors);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Give the added data files this data sequence number — the one of
    /// the snapshot the rewrite read its files at — instead of the new
    /// snapshot's. Their rows are the removed files' rows as of then, so
    /// equality deletes committed since still apply to them; with the new
    /// snapshot's number they would rise above those deletes and bring the
    /// deleted rows back. Java's `RewriteFiles.dataSequenceNumber`.
    pub fn set_data_sequence_number(mut self, data_sequence_number: i64) -> Self {
        self.data_sequence_number = Some(data_sequence_number);
        self
    }

    /// Fail the commit if a removed data file has row ids: for a rewrite
    /// whose added files store no row lineage, because it read its rows
    /// before they had ids — before the table's upgrade to v3, and the
    /// commit that then gave them theirs. Moved, they would take new ones,
    /// where the spec requires a moved row to keep its `_row_id`. Checked
    /// on every commit attempt, like the removed files being live.
    pub fn validate_no_row_ids(mut self) -> Self {
        self.validate_no_row_ids = true;
        self
    }
}

#[async_trait]
impl TransactionAction for RewriteFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // One read of the table this attempt would land on serves both
        // what the rewrite removes and which manifests it keeps.
        let manifests = current_manifests(table).await?;
        let removed_files = self.removed_files(&manifests)?;
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.added_delete_files.clone(),
        )
        .with_added_data_sequence_number(self.data_sequence_number)
        .with_removed_files(removed_files);

        snapshot_producer.validate_added_files(&self.added_data_files)?;
        snapshot_producer.validate_added_files(&self.added_delete_files)?;

        snapshot_producer
            .commit(
                RewriteFilesOperation {
                    removal: self.removal.clone(),
                    manifests,
                },
                DefaultManifestProcess,
            )
            .await
    }
}

/// Most manifests a rewrite reads at once.
const MANIFEST_READ_CONCURRENCY: usize = 16;

/// `table`'s current manifests, each with its manifest-list entry, in the
/// manifest list's order: read [`MANIFEST_READ_CONCURRENCY`] at a time, as
/// each is a round trip to the object store.
pub(super) async fn current_manifests(table: &Table) -> Result<Vec<(ManifestFile, Manifest)>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(Vec::new());
    };
    let manifest_list = table.manifest_list_reader(snapshot).load().await?;
    let entries: Vec<ManifestFile> = manifest_list.consume_entries().into_iter().collect();
    let runtime = table.runtime();
    futures::stream::iter(entries)
        .map(|entry| {
            let reader = table.manifest_reader();
            runtime.io().spawn(async move {
                let manifest = reader.read(&entry).await?;
                Ok::<_, Error>((entry, manifest))
            })
        })
        .buffered(MANIFEST_READ_CONCURRENCY)
        // The task's own result, inside its join's.
        .and_then(futures::future::ready)
        .try_collect()
        .await
}

/// The live files of `manifests`.
pub(super) fn live_files(
    manifests: &[(ManifestFile, Manifest)],
) -> impl Iterator<Item = &DataFile> {
    manifests.iter().flat_map(|(_, manifest)| {
        manifest
            .entries()
            .iter()
            .filter(|e| e.is_alive())
            .map(|e| e.data_file())
    })
}

/// Whether `file` is a deletion vector: a blob in a Puffin file, of which
/// the file may hold several.
pub(super) fn is_deletion_vector(file: &DataFile) -> bool {
    file.content_type() == DataContentType::PositionDeletes
        && file.file_format() == DataFileFormat::Puffin
}

/// The files a snapshot removes: by path, or a deletion vector by its
/// Puffin file's path and its blob's offset.
#[derive(Clone, Debug, Default)]
pub(super) struct Removal {
    pub(super) paths: HashSet<String>,
    pub(super) vectors: HashSet<(String, i64)>,
}

impl Removal {
    /// Whether it removes `file`.
    pub(super) fn contains(&self, file: &DataFile) -> bool {
        self.paths.contains(file.file_path())
            || file.content_offset().is_some_and(|offset| {
                self.vectors
                    .contains(&(file.file_path().to_string(), offset))
            })
    }

    /// The live files in `manifests` it removes — the snapshot summary
    /// subtracts them. Fails if one isn't live: another commit removed it
    /// first — saying so after `action` ("Rewrite"). Runs on every commit
    /// attempt, against the manifests of the table the commit would land
    /// on.
    pub(super) fn live_files(
        &self,
        manifests: &[(ManifestFile, Manifest)],
        action: &str,
    ) -> Result<Vec<DataFile>> {
        let mut removed = Vec::new();
        let mut paths: HashSet<&str> = HashSet::new();
        let mut vectors: HashSet<(&str, i64)> = HashSet::new();
        for file in live_files(manifests) {
            if self.contains(file) {
                paths.insert(file.file_path());
                if let Some(offset) = file.content_offset() {
                    vectors.insert((file.file_path(), offset));
                }
                removed.push(file.clone());
            }
        }
        let mut missing: Vec<String> = self
            .paths
            .iter()
            .filter(|p| !paths.contains(p.as_str()))
            .cloned()
            .chain(
                self.vectors
                    .iter()
                    .filter(|(p, o)| !vectors.contains(&(p.as_str(), *o)))
                    .map(|(p, o)| format!("{p}@{o}")),
            )
            .collect();
        if !missing.is_empty() {
            missing.sort();
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("{action} removes files no longer in the table: {missing:?}"),
            ));
        }
        Ok(removed)
    }
}

impl RewriteFilesAction {
    /// The live files in `manifests` that it removes — the snapshot
    /// summary subtracts them. Fails if one isn't live: another rewrite
    /// replaced it first, and committing this one's files as well would
    /// duplicate their rows. Fails too if a removed data file has a
    /// deletion vector it doesn't remove. Runs on every commit attempt,
    /// against the manifests of the table the commit would land on.
    fn removed_files(&self, manifests: &[(ManifestFile, Manifest)]) -> Result<Vec<DataFile>> {
        let removed = self.removal.live_files(manifests, "Rewrite")?;
        if self.validate_no_row_ids {
            let mut with_ids: Vec<&str> = removed
                .iter()
                .filter(|f| f.content_type() == DataContentType::Data && f.first_row_id().is_some())
                .map(|f| f.file_path())
                .collect();
            if !with_ids.is_empty() {
                with_ids.sort();
                return Err(Error::new(
                    ErrorKind::CatalogCommitConflicts,
                    format!("Rewrite moves rows that have row ids, without them: {with_ids:?}"),
                ));
            }
        }
        let removed_data: HashSet<&str> = removed
            .iter()
            .filter(|f| f.content_type() == DataContentType::Data)
            .map(|f| f.file_path())
            .collect();
        let mut kept: Vec<String> = live_files(manifests)
            .filter(|f| is_deletion_vector(f) && !self.removal.contains(f))
            .filter_map(|f| f.referenced_data_file())
            .filter(|data| removed_data.contains(data.as_str()))
            .collect();
        if !kept.is_empty() {
            kept.sort();
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!(
                    "Rewrite removes data files whose deletion vectors it keeps — rows deleted \
                     since it read them: {kept:?}"
                ),
            ));
        }
        Ok(removed)
    }
}

struct RewriteFilesOperation {
    removal: Removal,
    /// The current snapshot's manifests, as the commit read them.
    manifests: Vec<(ManifestFile, Manifest)>,
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
        carry_manifests(snapshot_produce, &self.manifests, &self.removal).await
    }
}

/// The manifests a snapshot that removes `removal`'s files carries from
/// `manifests`, the current snapshot's: each that holds none of them
/// unchanged; none that holds nothing else live; and the rest rewritten,
/// their other live entries as `Existing` (each keeping its original
/// `snapshot_id` and `sequence_number`). The entries earlier commits left
/// `DELETED` — Java writes them for what it removes — aren't carried:
/// rewritten as `Existing`, their files would be live again.
pub(super) async fn carry_manifests(
    snapshot_produce: &mut SnapshotProducer<'_>,
    manifests: &[(ManifestFile, Manifest)],
    removal: &Removal,
) -> Result<Vec<ManifestFile>> {
    // Pass 1: classify each manifest based on its loaded entries. We
    // can't write rewritten manifests yet because that needs `&mut
    // snapshot_produce`, and we also need the producer to compute
    // file-IO paths during this loop.
    enum Plan {
        CarryForward(ManifestFile),
        DropEntirely,
        Rewrite {
            survivors: Vec<ManifestEntry>,
            content: crate::spec::ManifestContentType,
        },
    }
    let mut plans: Vec<Plan> = Vec::with_capacity(manifests.len());
    for (mfile, manifest) in manifests {
        let alive = manifest.entries().iter().filter(|me| me.is_alive());
        let removes_any = alive.clone().any(|me| removal.contains(me.data_file()));
        let survivors: Vec<ManifestEntry> = alive
            .filter(|me| !removal.contains(me.data_file()))
            .map(|me| ManifestEntry::clone(me))
            .collect();

        if !removes_any {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, ManifestContentType, NestedField,
        Operation, PrimitiveType, Schema, Struct, Type,
    };
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, ErrorKind};

    async fn make_memory_table() -> (crate::memory::MemoryCatalog, crate::table::Table) {
        make_memory_table_in(crate::spec::FormatVersion::V2).await
    }

    async fn make_memory_table_in(
        format_version: crate::spec::FormatVersion,
    ) -> (crate::memory::MemoryCatalog, crate::table::Table) {
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
            .format_version(format_version)
            .build();
        let table = memory.create_table(&ns, creation).await.unwrap();
        (memory, table)
    }

    /// Each live data file's first row id in `table`'s current snapshot.
    async fn first_row_ids(table: &crate::table::Table) -> HashMap<String, Option<i64>> {
        let snap = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snap).load().await.unwrap();
        let mut out = HashMap::new();
        for m in list.entries() {
            let manifest = table.manifest_reader().read(m).await.unwrap();
            for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                let f = e.data_file();
                out.insert(f.file_path().to_string(), f.first_row_id());
            }
        }
        out
    }

    /// On a v3 table, the files a rewrite leaves in a manifest it rewrites
    /// keep their first row ids — their rows keep their `_row_id`s.
    #[tokio::test]
    async fn rewrite_keeps_the_row_ids_of_the_files_it_leaves() {
        let (catalog, table) = make_memory_table_in(crate::spec::FormatVersion::V3).await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone(), f2.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        let before = first_row_ids(&table).await;
        assert_eq!(before[f2.file_path()], Some(10));

        let compacted = data_file("memory:///warehouse/public/orders/data/c.parquet", 10, 512);
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted.clone()])
            .remove_data_files(vec![f1.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        let after = first_row_ids(&table).await;
        assert_eq!(after[f2.file_path()], Some(10), "{after:?}");
        // Fresh ids, above every earlier one. (The rewritten manifest takes
        // a range for the file it carries too, as Java's writer assigns one.)
        assert!(after[compacted.file_path()] >= Some(30), "{after:?}");
    }

    /// A rewrite told its added files store no row lineage fails on files
    /// whose rows have ids, and goes through on files whose rows don't.
    #[tokio::test]
    async fn rewrite_validating_no_row_ids_fails_on_files_with_them() {
        for (format_version, fails) in [
            (crate::spec::FormatVersion::V2, false),
            (crate::spec::FormatVersion::V3, true),
        ] {
            let (catalog, table) = make_memory_table_in(format_version).await;
            let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
            let tx = Transaction::new(&table);
            let action = tx
                .fast_append()
                .with_check_duplicate(false)
                .add_data_files(vec![f1.clone()]);
            let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();

            let compacted = data_file("memory:///warehouse/public/orders/data/c.parquet", 10, 512);
            let tx = Transaction::new(&table);
            let action = tx
                .rewrite_files()
                .add_data_files(vec![compacted])
                .remove_data_files(vec![f1])
                .validate_no_row_ids();
            let result = action.apply(tx).unwrap().commit(&catalog).await;
            match result {
                Err(e) if fails => assert_eq!(e.kind(), ErrorKind::CatalogCommitConflicts),
                Ok(_) if !fails => {}
                other => panic!("{format_version:?}: {:?}", other.map(|_| ())),
            }
        }
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

        let mlist = table.manifest_list_reader(snap).load().await.unwrap();
        let data_manifests: Vec<_> = mlist
            .entries()
            .iter()
            .filter(|m| m.content == ManifestContentType::Data)
            .collect();
        assert_eq!(data_manifests.len(), 1);

        let manifest = table
            .manifest_reader()
            .read(data_manifests[0])
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

        let mlist = table.manifest_list_reader(snap).load().await.unwrap();
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
            let m = table.manifest_reader().read(mf).await.unwrap();
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

        let d1 = data_file(
            "memory:///warehouse/public/orders/data/d1.parquet",
            10,
            1024,
        );
        let d2 = data_file(
            "memory:///warehouse/public/orders/data/d2.parquet",
            20,
            2048,
        );
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
        let mlist = table.manifest_list_reader(snap).load().await.unwrap();

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
            let m = table.manifest_reader().read(mf).await.unwrap();
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

        let mlist = table.manifest_list_reader(snap).load().await.unwrap();
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

    /// With a data sequence number, the added files carry it rather than
    /// the new snapshot's, which still becomes their file sequence number.
    #[tokio::test]
    async fn rewrite_gives_added_files_its_data_sequence_number() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let mut table = table;
        for f in [&f1, &f2] {
            let tx = Transaction::new(&table);
            let action = tx
                .fast_append()
                .with_check_duplicate(false)
                .add_data_files(vec![f.clone()]);
            table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        }

        // Rewrite f1 as planned at sequence number 1.
        let compacted = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            10,
            1024,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted.clone()])
            .remove_data_files(vec![f1.clone()])
            .set_data_sequence_number(1);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        assert_eq!(snap.sequence_number(), 3);
        let mlist = table.manifest_list_reader(snap).load().await.unwrap();
        let mut seqs = Vec::new();
        for mf in mlist.entries() {
            let m = table.manifest_reader().read(mf).await.unwrap();
            for entry in m.entries() {
                seqs.push((
                    entry.data_file().file_path().to_string(),
                    entry.sequence_number(),
                    entry.file_sequence_number,
                ));
            }
        }
        seqs.sort();
        assert_eq!(seqs, vec![
            (f2.file_path().to_string(), Some(2), Some(2)),
            (compacted.file_path().to_string(), Some(1), Some(3)),
        ]);
    }

    /// A rewrite of a file another rewrite already removed fails: adding
    /// its files too would duplicate the rows.
    #[tokio::test]
    async fn rewrite_of_a_file_no_longer_in_the_table_fails() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![f1.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();

        // Two rewrites of f1, planned against the same table.
        let rewrite = |path: &str| {
            let tx = Transaction::new(&table);
            let action = tx
                .rewrite_files()
                .add_data_files(vec![data_file(path, 10, 1024)])
                .remove_data_files(vec![f1.clone()]);
            action.apply(tx).unwrap()
        };
        let first = rewrite("memory:///warehouse/public/orders/data/compact-a.parquet");
        let second = rewrite("memory:///warehouse/public/orders/data/compact-b.parquet");
        let table = first.commit(&catalog).await.unwrap();
        // The second lands on the first's table, without f1.
        let err = second.commit(&catalog).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
        assert!(!err.retryable());

        let table = catalog.load_table(table.identifier()).await.unwrap();
        let snap = table.metadata().current_snapshot().unwrap();
        let mlist = table.manifest_list_reader(snap).load().await.unwrap();
        let mut paths = Vec::new();
        for mf in mlist.entries() {
            let m = table.manifest_reader().read(mf).await.unwrap();
            for entry in m.entries() {
                paths.push(entry.data_file().file_path().to_string());
            }
        }
        assert_eq!(paths, vec![
            "memory:///warehouse/public/orders/data/compact-a.parquet".to_string()
        ]);
    }

    /// So does one removing a file from a table with no snapshot at all.
    #[tokio::test]
    async fn rewrite_removing_from_an_empty_table_fails() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let compacted = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            10,
            1024,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted])
            .remove_data_files(vec![f1]);
        let err = action
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
    }

    /// The snapshot summary keeps running totals: a rewrite subtracts the
    /// files it removes, data and deletes, and adds its outputs.
    #[tokio::test]
    async fn rewrite_summary_keeps_running_totals() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
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
        let mut table = table;
        for files in [vec![f1.clone()], vec![f2.clone(), eq_delete.clone()]] {
            let tx = Transaction::new(&table);
            let action = tx
                .fast_append()
                .with_check_duplicate(false)
                .add_data_files(files);
            table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        }
        let summary = |table: &crate::table::Table, key: &str| -> String {
            let snap = table.metadata().current_snapshot().unwrap();
            snap.summary().additional_properties[key].clone()
        };
        assert_eq!(summary(&table, "total-records"), "30");
        assert_eq!(summary(&table, "total-delete-files"), "1");

        // f1 rewritten as 9 rows, retiring the delete.
        let compacted = data_file(
            "memory:///warehouse/public/orders/data/compact-0.parquet",
            9,
            1024,
        );
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted])
            .remove_data_files(vec![f1.clone(), eq_delete.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();

        assert_eq!(summary(&table, "deleted-data-files"), "1");
        assert_eq!(summary(&table, "deleted-records"), "10");
        assert_eq!(summary(&table, "removed-delete-files"), "1");
        assert_eq!(summary(&table, "total-records"), "29");
        assert_eq!(summary(&table, "total-data-files"), "2");
        assert_eq!(summary(&table, "total-delete-files"), "0");
        assert_eq!(summary(&table, "total-equality-deletes"), "0");
    }

    /// Another engine's commit, as Java writes one: the manifest holding
    /// a file it removes, rewritten with that file `DELETED`.
    struct JavaStyleRemoval {
        removed: String,
    }

    struct WithManifest(crate::spec::ManifestFile);

    impl crate::transaction::snapshot::SnapshotProduceOperation for WithManifest {
        fn operation(&self) -> Operation {
            Operation::Delete
        }

        async fn delete_entries(
            &self,
            _: &crate::transaction::snapshot::SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestEntry>> {
            Ok(vec![])
        }

        async fn existing_manifest(
            &self,
            _: &mut crate::transaction::snapshot::SnapshotProducer<'_>,
        ) -> crate::Result<Vec<crate::spec::ManifestFile>> {
            Ok(vec![self.0.clone()])
        }
    }

    #[async_trait::async_trait]
    impl crate::transaction::TransactionAction for JavaStyleRemoval {
        async fn commit(
            self: std::sync::Arc<Self>,
            table: &crate::table::Table,
        ) -> crate::Result<crate::transaction::ActionCommit> {
            let snap = table.metadata().current_snapshot().unwrap();
            let list = table.manifest_list_reader(snap).load().await.unwrap();
            assert_eq!(list.entries().len(), 1);
            let manifest = table
                .manifest_reader()
                .read(&list.entries()[0])
                .await
                .unwrap();
            let mut writer = crate::spec::ManifestWriterBuilder::new(
                table.file_io().new_output(format!(
                    "{}/metadata/java-m0.avro",
                    table.metadata().location()
                ))?,
                Some(snap.snapshot_id()),
                table.metadata().current_schema().clone(),
                table.metadata().default_partition_spec().as_ref().clone(),
            )
            .build_v2_data();
            for e in manifest.entries() {
                let file = e.data_file().clone();
                let seq = e.sequence_number().unwrap();
                if file.file_path() == self.removed {
                    writer.add_delete_file(file, seq, e.file_sequence_number)?;
                } else {
                    writer.add_existing_file(
                        file,
                        e.snapshot_id().unwrap(),
                        seq,
                        e.file_sequence_number,
                    )?;
                }
            }
            let mut rewritten = writer.write_manifest_file().await?;
            // Written by the new snapshot, whose id the producer picks.
            rewritten.sequence_number = table.metadata().next_sequence_number();
            rewritten.min_sequence_number = snap.sequence_number();
            crate::transaction::snapshot::SnapshotProducer::new(
                table,
                uuid::Uuid::now_v7(),
                HashMap::from([("engine".to_string(), "java".to_string())]),
                vec![],
                vec![],
            )
            .commit(
                WithManifest(rewritten),
                crate::transaction::snapshot::DefaultManifestProcess,
            )
            .await
        }
    }

    /// A manifest another engine rewrote holds what it removed as
    /// `DELETED` entries. A rewrite removing another of its files doesn't
    /// carry those as `Existing`: they'd be live again.
    #[tokio::test]
    async fn rewrite_doesnt_bring_back_what_an_earlier_commit_removed() {
        let (catalog, table) = make_memory_table().await;
        let f1 = data_file("memory:///warehouse/public/orders/data/1.parquet", 10, 1024);
        let f2 = data_file("memory:///warehouse/public/orders/data/2.parquet", 20, 2048);
        let f3 = data_file("memory:///warehouse/public/orders/data/3.parquet", 30, 3072);
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![f1.clone(), f2.clone(), f3.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        let tx = Transaction::new(&table);
        let removal = JavaStyleRemoval {
            removed: f1.file_path().to_string(),
        };
        let table = removal.apply(tx).unwrap().commit(&catalog).await.unwrap();

        let compacted = data_file("memory:///warehouse/public/orders/data/c.parquet", 20, 512);
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![compacted.clone()])
            .remove_data_files(vec![f2.clone()]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();

        let snap = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snap).load().await.unwrap();
        let mut live = Vec::new();
        for m in list.entries() {
            let manifest = table.manifest_reader().read(m).await.unwrap();
            live.extend(
                manifest
                    .entries()
                    .iter()
                    .filter(|e| e.is_alive())
                    .map(|e| e.data_file().file_path().to_string()),
            );
        }
        live.sort();
        assert_eq!(live, vec![
            f3.file_path().to_string(),
            compacted.file_path().to_string()
        ]);
    }
}
