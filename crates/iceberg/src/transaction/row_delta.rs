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

use crate::error::{Error, ErrorKind, Result};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, Manifest, ManifestEntry, ManifestFile, Operation,
};
use crate::table::Table;
use crate::transaction::rewrite::{
    Removal, carry_manifests, current_manifests, is_deletion_vector, live_files,
};
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// `RowDeltaAction` produces a snapshot that changes rows: it adds data
/// files and delete files — on a v3 table, deletion vectors — and removes
/// delete files, in one commit. Java's `RowDelta`. Its operation is
/// `delete` if it adds only deletes, and `overwrite` otherwise.
///
/// A table holds at most one deletion vector per data file: one deleting
/// more of a data file's rows holds its vector's positions and the new
/// ones, and the action removes the vector it replaces
/// ([`Self::remove_deletion_vectors`]).
///
/// Checked on every commit attempt, against the table the commit would
/// land on, failing with [`ErrorKind::CatalogCommitConflicts`] — so the
/// caller plans again:
/// - each delete file it removes is still live;
/// - each deletion vector it adds names a data file that is live, or that
///   it adds — a concurrent rewrite may have removed it — and that has no
///   other vector: none live it doesn't remove, none other it adds.
///
/// On a v3 table every position delete it adds must be a deletion vector,
/// and before v3, none may be, as the spec requires.
pub struct RowDeltaAction {
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
    removal: Removal,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
            removal: Removal::default(),
        }
    }

    /// Add files: routed by `content_type()` into the data manifest or the
    /// delete manifest.
    pub fn add_data_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        for file in files {
            match file.content_type() {
                DataContentType::Data => self.added_data_files.push(file),
                DataContentType::EqualityDeletes | DataContentType::PositionDeletes => {
                    self.added_delete_files.push(file)
                }
            }
        }
        self
    }

    /// Remove deletion vectors, each by its Puffin file's path and its
    /// blob's offset there (`content_offset`): the ones those it adds
    /// replace.
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

    /// Fail unless every position delete it adds is a deletion vector on a
    /// v3 table, and none is before v3.
    fn validate_delete_formats(&self, format_version: FormatVersion) -> Result<()> {
        for file in &self.added_delete_files {
            if file.content_type() != DataContentType::PositionDeletes {
                continue;
            }
            let vector = is_deletion_vector(file);
            match format_version {
                FormatVersion::V3 if !vector => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Must use deletion vectors for position deletes in v3: {}",
                            file.file_path()
                        ),
                    ));
                }
                FormatVersion::V3 => {
                    let complete = file.referenced_data_file().is_some()
                        && file.content_offset().is_some()
                        && file.content_size_in_bytes().is_some();
                    if !complete {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "A deletion vector needs its data file, offset and size: {}",
                                file.file_path()
                            ),
                        ));
                    }
                }
                FormatVersion::V1 | FormatVersion::V2 if vector => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Must not use deletion vectors before v3: {}",
                            file.file_path()
                        ),
                    ));
                }
                FormatVersion::V1 | FormatVersion::V2 => {}
            }
        }
        Ok(())
    }

    /// Fail unless each deletion vector it adds names a data file live in
    /// `manifests` or added here, which has no other vector once this
    /// commit lands.
    fn validate_deletion_vectors(&self, manifests: &[(ManifestFile, Manifest)]) -> Result<()> {
        let added: Vec<&DataFile> = self
            .added_delete_files
            .iter()
            .filter(|f| is_deletion_vector(f))
            .collect();
        if added.is_empty() {
            return Ok(());
        }
        let mut data: HashSet<&str> = self
            .added_data_files
            .iter()
            .map(|f| f.file_path())
            .collect();
        let mut vectors: HashSet<String> = HashSet::new();
        for file in live_files(manifests) {
            match file.content_type() {
                DataContentType::Data => {
                    data.insert(file.file_path());
                }
                _ if is_deletion_vector(file) && !self.removal.contains(file) => {
                    vectors.extend(file.referenced_data_file());
                }
                _ => {}
            }
        }
        let mut missing = Vec::new();
        let mut doubled = Vec::new();
        for file in added {
            let Some(referenced) = file.referenced_data_file() else {
                continue;
            };
            if !data.contains(referenced.as_str()) {
                missing.push(referenced);
            } else if !vectors.insert(referenced.clone()) {
                doubled.push(referenced);
            }
        }
        if !missing.is_empty() {
            missing.sort();
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("Deletion vectors name data files no longer in the table: {missing:?}"),
            ));
        }
        if !doubled.is_empty() {
            doubled.sort();
            return Err(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!("Data files would have two deletion vectors: {doubled:?}"),
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl TransactionAction for RowDeltaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        self.validate_delete_formats(table.metadata().format_version())?;
        let manifests = current_manifests(table).await?;
        let removed_files = self.removal.live_files(&manifests, "Row delta")?;
        self.validate_deletion_vectors(&manifests)?;
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.added_delete_files.clone(),
        )
        .with_removed_files(removed_files);

        snapshot_producer.validate_added_files(&self.added_data_files)?;
        snapshot_producer.validate_added_files(&self.added_delete_files)?;

        let operation = if self.added_data_files.is_empty() && !self.added_delete_files.is_empty() {
            Operation::Delete
        } else {
            Operation::Overwrite
        };
        snapshot_producer
            .commit(
                RowDeltaOperation {
                    operation,
                    removal: self.removal.clone(),
                    manifests,
                },
                DefaultManifestProcess,
            )
            .await
    }
}

struct RowDeltaOperation {
    operation: Operation,
    removal: Removal,
    /// The current snapshot's manifests, as the commit read them.
    manifests: Vec<(ManifestFile, Manifest)>,
}

impl SnapshotProduceOperation for RowDeltaOperation {
    fn operation(&self) -> Operation {
        self.operation.clone()
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        carry_manifests(snapshot_produce, &self.manifests, &self.removal).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, NestedField,
        Operation, PrimitiveType, Schema, Struct, Type,
    };
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, CatalogBuilder, ErrorKind};

    const DIR: &str = "memory:///warehouse/public/orders";

    async fn table_in(format_version: FormatVersion) -> (crate::memory::MemoryCatalog, Table) {
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids([1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
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

    fn data(name: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{DIR}/data/{name}"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1024)
            .record_count(10)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// A vector in Puffin file `puffin`, at `offset`, deleting `deleted`
    /// rows of data file `of`.
    fn vector(puffin: &str, offset: i64, of: &str, deleted: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(format!("{DIR}/data/{puffin}"))
            .file_format(DataFileFormat::Puffin)
            .file_size_in_bytes(4096)
            .record_count(deleted)
            .referenced_data_file(Some(format!("{DIR}/data/{of}")))
            .content_offset(Some(offset))
            .content_size_in_bytes(Some(40))
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    fn place(file: &DataFile) -> (String, i64) {
        (file.file_path().to_string(), file.content_offset().unwrap())
    }

    /// The live files of `table`'s current snapshot: data files by path,
    /// vectors by place.
    async fn live(table: &Table) -> BTreeSet<String> {
        let snap = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snap).load().await.unwrap();
        let mut out = BTreeSet::new();
        for m in list.entries() {
            let manifest = table.manifest_reader().read(m).await.unwrap();
            for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                let f = e.data_file();
                let name = f.file_path().rsplit('/').next().unwrap().to_string();
                out.insert(match f.content_offset() {
                    Some(offset) => format!("{name}@{offset}"),
                    None => name,
                });
            }
        }
        out
    }

    fn names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    async fn append(
        catalog: &crate::memory::MemoryCatalog,
        table: &Table,
        files: Vec<DataFile>,
    ) -> Table {
        let tx = Transaction::new(table);
        let action = tx.fast_append().add_data_files(files);
        action.apply(tx).unwrap().commit(catalog).await.unwrap()
    }

    async fn delta(
        catalog: &crate::memory::MemoryCatalog,
        table: &Table,
        files: Vec<DataFile>,
        replaced: Vec<&DataFile>,
    ) -> crate::Result<Table> {
        let tx = Transaction::new(table);
        let action = tx
            .row_delta()
            .add_data_files(files)
            .remove_deletion_vectors(replaced.into_iter().map(place));
        action.apply(tx).unwrap().commit(catalog).await
    }

    /// The manifests of `table`'s current snapshot holding deletes.
    async fn delete_manifests(table: &Table) -> usize {
        let snap = table.metadata().current_snapshot().unwrap();
        let list = table.manifest_list_reader(snap).load().await.unwrap();
        list.entries()
            .iter()
            .filter(|m| m.content == crate::spec::ManifestContentType::Deletes)
            .count()
    }

    /// A commit replacing vectors of several manifests keeps the vectors
    /// they hold that it doesn't replace in one manifest, beside the one it
    /// adds. Rewritten one each, a table's delete manifests would multiply
    /// with its commits, and each commit read and wrote more of them.
    #[tokio::test]
    async fn a_row_delta_keeps_the_survivors_of_the_manifests_it_touches_in_one() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let files = ["a", "b", "c", "d"].map(|n| data(&format!("{n}.parquet")));
        let table = append(&catalog, &table, files.to_vec()).await;
        let of_a = vector("p1.puffin", 4, "a.parquet", 1);
        let of_b = vector("p1.puffin", 44, "b.parquet", 1);
        let table = delta(&catalog, &table, vec![of_a.clone(), of_b], vec![])
            .await
            .unwrap();
        let of_c = vector("p2.puffin", 4, "c.parquet", 1);
        let of_d = vector("p2.puffin", 44, "d.parquet", 1);
        let table = delta(&catalog, &table, vec![of_c.clone(), of_d], vec![])
            .await
            .unwrap();
        let replacing = vec![
            vector("p3.puffin", 4, "a.parquet", 2),
            vector("p3.puffin", 44, "c.parquet", 2),
        ];
        let table = delta(&catalog, &table, replacing, vec![&of_a, &of_c])
            .await
            .unwrap();
        assert_eq!(
            live(&table).await,
            names(&[
                "a.parquet",
                "b.parquet",
                "c.parquet",
                "d.parquet",
                "p1.puffin@44",
                "p2.puffin@44",
                "p3.puffin@4",
                "p3.puffin@44",
            ])
        );
        assert_eq!(delete_manifests(&table).await, 2);
    }

    fn summary(table: &Table, key: &str) -> Option<String> {
        let snap = table.metadata().current_snapshot().unwrap();
        snap.summary().additional_properties.get(key).cloned()
    }

    #[tokio::test]
    async fn a_row_delta_adds_rows_and_deletes_them() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        // An update: a row of `a` deleted, its new version added.
        let table = delta(
            &catalog,
            &table,
            vec![data("b.parquet"), vector("p1.puffin", 4, "a.parquet", 1)],
            vec![],
        )
        .await
        .unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Overwrite
        );
        assert_eq!(
            live(&table).await,
            names(&["a.parquet", "b.parquet", "p1.puffin@4"])
        );
        assert_eq!(summary(&table, "added-dvs").as_deref(), Some("1"));
        assert_eq!(summary(&table, "total-delete-files").as_deref(), Some("1"));
        assert_eq!(
            summary(&table, "total-position-deletes").as_deref(),
            Some("1")
        );
        assert_eq!(summary(&table, "total-data-files").as_deref(), Some("2"));
        // A vector counts at its blob's size.
        assert_eq!(summary(&table, "total-files-size").as_deref(), Some("2088"));

        // Deletes alone are a `delete`.
        let a = vector("p1.puffin", 4, "a.parquet", 1);
        let table = delta(
            &catalog,
            &table,
            vec![vector("p2.puffin", 4, "a.parquet", 2)],
            vec![&a],
        )
        .await
        .unwrap();
        assert_eq!(
            table
                .metadata()
                .current_snapshot()
                .unwrap()
                .summary()
                .operation,
            Operation::Delete
        );
        assert_eq!(summary(&table, "removed-dvs").as_deref(), Some("1"));
        assert_eq!(summary(&table, "total-delete-files").as_deref(), Some("1"));
        assert_eq!(
            summary(&table, "total-position-deletes").as_deref(),
            Some("2")
        );
    }

    /// A Puffin file holds several vectors: replacing one keeps the others.
    #[tokio::test]
    async fn a_vector_replaces_its_data_files_last_by_its_place() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet"), data("b.parquet")]).await;
        let of_a = vector("p1.puffin", 4, "a.parquet", 1);
        let of_b = vector("p1.puffin", 44, "b.parquet", 1);
        let table = delta(&catalog, &table, vec![of_a.clone(), of_b.clone()], vec![])
            .await
            .unwrap();
        let table = delta(
            &catalog,
            &table,
            vec![vector("p2.puffin", 4, "a.parquet", 2)],
            vec![&of_a],
        )
        .await
        .unwrap();
        assert_eq!(
            live(&table).await,
            names(&["a.parquet", "b.parquet", "p1.puffin@44", "p2.puffin@4"])
        );
    }

    #[tokio::test]
    async fn a_second_vector_for_a_data_file_conflicts() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        let first = vector("p1.puffin", 4, "a.parquet", 1);
        let table = delta(&catalog, &table, vec![first], vec![]).await.unwrap();
        let err = delta(
            &catalog,
            &table,
            vec![vector("p2.puffin", 4, "a.parquet", 2)],
            vec![],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
        // Nor two in one commit.
        let err = delta(
            &catalog,
            &table,
            vec![
                data("b.parquet"),
                vector("p2.puffin", 4, "b.parquet", 1),
                vector("p2.puffin", 44, "b.parquet", 1),
            ],
            vec![],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
    }

    /// A vector replaced meanwhile — by another writer's — can't be again.
    #[tokio::test]
    async fn replacing_a_vector_no_longer_live_conflicts() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        let first = vector("p1.puffin", 4, "a.parquet", 1);
        let table = delta(&catalog, &table, vec![first.clone()], vec![])
            .await
            .unwrap();
        let stale = table.clone();
        delta(
            &catalog,
            &table,
            vec![vector("p2.puffin", 4, "a.parquet", 2)],
            vec![&first],
        )
        .await
        .unwrap();
        // Planned on the table before, committed on the table after.
        let err = delta(
            &catalog,
            &stale,
            vec![vector("p3.puffin", 4, "a.parquet", 2)],
            vec![&first],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
    }

    /// A vector removed meanwhile — its rows undeleted — can't be replaced
    /// with one holding its positions.
    #[tokio::test]
    async fn replacing_a_vector_removed_meanwhile_conflicts() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        let first = vector("p1.puffin", 4, "a.parquet", 1);
        let table = delta(&catalog, &table, vec![first.clone()], vec![])
            .await
            .unwrap();
        let stale = table.clone();
        let tx = Transaction::new(&table);
        let action = tx
            .row_delta()
            .remove_deletion_vectors([place(&first)])
            .set_snapshot_properties(HashMap::from([("undelete".into(), "a".into())]));
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        assert_eq!(live(&table).await, names(&["a.parquet"]));
        let err = delta(
            &catalog,
            &stale,
            vec![vector("p2.puffin", 4, "a.parquet", 2)],
            vec![&first],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
    }

    /// A rewrite removed the data file a vector names: its rows are in the
    /// rewrite's files now.
    #[tokio::test]
    async fn a_vector_for_a_data_file_no_longer_live_conflicts() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        let stale = table.clone();
        let tx = Transaction::new(&table);
        let action = tx
            .rewrite_files()
            .add_data_files(vec![data("c.parquet")])
            .remove_data_files(vec![data("a.parquet")]);
        action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        let err = delta(
            &catalog,
            &stale,
            vec![vector("p1.puffin", 4, "a.parquet", 1)],
            vec![],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
    }

    #[tokio::test]
    async fn position_deletes_are_vectors_on_v3_only() {
        let (catalog, table) = table_in(FormatVersion::V2).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        let err = delta(
            &catalog,
            &table,
            vec![vector("p1.puffin", 4, "a.parquet", 1)],
            vec![],
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid, "{err}");

        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet")]).await;
        // Naming its data file, offset and size as a vector does.
        let position_deletes = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(format!("{DIR}/data/deletes.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .referenced_data_file(Some(format!("{DIR}/data/a.parquet")))
            .content_offset(Some(4))
            .content_size_in_bytes(Some(40))
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap();
        let err = delta(&catalog, &table, vec![position_deletes], vec![])
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::DataInvalid, "{err}");
    }

    /// A rewrite takes the vectors of the data files it removes with them,
    /// by their place, and fails on one it doesn't: rows deleted after it
    /// read the file, still in its output.
    #[tokio::test]
    async fn a_rewrite_removes_its_inputs_vectors_or_fails() {
        let (catalog, table) = table_in(FormatVersion::V3).await;
        let table = append(&catalog, &table, vec![data("a.parquet"), data("b.parquet")]).await;
        let of_a = vector("p1.puffin", 4, "a.parquet", 1);
        let of_b = vector("p1.puffin", 44, "b.parquet", 1);
        let table = delta(&catalog, &table, vec![of_a.clone(), of_b], vec![])
            .await
            .unwrap();
        let rewrite = |vectors: Vec<(String, i64)>| {
            let tx = Transaction::new(&table);
            let action = tx
                .rewrite_files()
                .add_data_files(vec![data("c.parquet")])
                .remove_data_files(vec![data("a.parquet")])
                .remove_deletion_vectors(vectors);
            (tx, action)
        };
        let (tx, action) = rewrite(vec![]);
        let err = action
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts, "{err}");
        let (tx, action) = rewrite(vec![place(&of_a)]);
        let table = action.apply(tx).unwrap().commit(&catalog).await.unwrap();
        assert_eq!(
            live(&table).await,
            names(&["b.parquet", "c.parquet", "p1.puffin@44"])
        );
    }
}
