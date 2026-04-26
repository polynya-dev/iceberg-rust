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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// FastAppendAction is a transaction action that appends data files and/or
/// equality-/position-delete files to the table in a single snapshot.
///
/// Files passed via [`Self::add_data_files`] are routed by `content_type()`
/// into separate manifests at commit time — `Data` files into a data
/// manifest, `EqualityDeletes`/`PositionDeletes` into a delete manifest. The
/// Iceberg spec forbids mixing content types in one manifest.
pub struct FastAppendAction {
    check_duplicate: bool,
    // below are properties used to create SnapshotProducer when commit
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
}

impl FastAppendAction {
    pub(crate) fn new() -> Self {
        Self {
            check_duplicate: true,
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
        }
    }

    /// Set whether to check duplicate files
    pub fn with_check_duplicate(mut self, v: bool) -> Self {
        self.check_duplicate = v;
        self
    }

    /// Add files to the snapshot. Files are routed by `content_type()`
    /// into the data manifest (for `Data`) or the delete manifest (for
    /// `EqualityDeletes` / `PositionDeletes`).
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        for file in data_files {
            match file.content_type() {
                crate::spec::DataContentType::Data => self.added_data_files.push(file),
                crate::spec::DataContentType::EqualityDeletes
                | crate::spec::DataContentType::PositionDeletes => {
                    self.added_delete_files.push(file)
                }
            }
        }
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
impl TransactionAction for FastAppendAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.added_delete_files.clone(),
        );

        // Both vecs follow the same partition-spec / partition-value rules.
        snapshot_producer.validate_added_files(&self.added_data_files)?;
        snapshot_producer.validate_added_files(&self.added_delete_files)?;

        // Checks duplicate files
        if self.check_duplicate {
            snapshot_producer.validate_duplicate_files().await?;
        }

        snapshot_producer
            .commit(FastAppendOperation, DefaultManifestProcess)
            .await
    }
}

struct FastAppendOperation;

impl SnapshotProduceOperation for FastAppendOperation {
    fn operation(&self) -> Operation {
        Operation::Append
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
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        Ok(manifest_list
            .entries()
            .iter()
            .filter(|entry| entry.has_added_files() || entry.has_existing_files())
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH, Struct,
    };
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};
    use crate::{TableRequirement, TableUpdate};

    #[tokio::test]
    async fn test_empty_data_append_action() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_set_snapshot_properties() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut snapshot_properties = HashMap::new();
        snapshot_properties.insert("key".to_string(), "val".to_string());

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = tx
            .fast_append()
            .set_snapshot_properties(snapshot_properties)
            .add_data_files(vec![data_file]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        // Check customized properties is contained in snapshot summary properties.
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot
                .summary()
                .additional_properties
                .get("key")
                .unwrap(),
            "val"
        );
    }

    #[tokio::test]
    async fn test_append_snapshot_properties() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut snapshot_properties = HashMap::new();
        snapshot_properties.insert("key".to_string(), "val".to_string());

        let action = tx
            .fast_append()
            .set_snapshot_properties(snapshot_properties);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        // Check customized properties is contained in snapshot summary properties.
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot
                .summary()
                .additional_properties
                .get("key")
                .unwrap(),
            "val"
        );
    }

    /// FastAppend now accepts mixed data + equality-delete files and routes
    /// them into separate manifests at commit time. The Iceberg manifest list
    /// should end up with two entries — one with `ManifestContentType::Data`,
    /// one with `Deletes`.
    #[tokio::test]
    async fn test_fast_append_routes_equality_deletes_into_separate_manifest() {
        use crate::spec::{ManifestContentType, NestedField, PrimitiveType, Schema, Type};

        // Build a minimal in-memory table with FileIO::new_with_memory so the
        // manifest writer can actually persist files.
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids([1])
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "qty", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();

        use crate::CatalogBuilder;
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
        memory
            .create_namespace(&ns, HashMap::new())
            .await
            .unwrap();
        let creation = crate::TableCreation::builder()
            .name("orders".to_string())
            .schema(schema)
            .build();
        let table = memory.create_table(&ns, creation).await.unwrap();

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(format!("{}/data/0001.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(2048)
            .record_count(7)
            .partition(Struct::empty())
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .build()
            .unwrap();

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(format!("{}/deletes/0001.parquet", table.metadata().location()))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(64)
            .record_count(1)
            .equality_ids(Some(vec![1]))
            .partition(Struct::empty())
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .build()
            .unwrap();

        use crate::transaction::ApplyTransactionAction;
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![data_file.clone(), delete_file.clone()]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&memory).await.unwrap();

        let snap = table
            .metadata()
            .current_snapshot()
            .expect("commit must produce a snapshot");
        let manifest_list = snap
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        // Two manifests — one Data, one Deletes.
        assert_eq!(manifest_list.entries().len(), 2);
        let mut content_types: Vec<ManifestContentType> =
            manifest_list.entries().iter().map(|e| e.content).collect();
        content_types.sort_by_key(|c| match c {
            ManifestContentType::Data => 0,
            ManifestContentType::Deletes => 1,
        });
        assert_eq!(content_types, vec![ManifestContentType::Data, ManifestContentType::Deletes]);

        // Walk each manifest and confirm the right file landed in the right
        // bucket.
        for entry in manifest_list.entries() {
            let manifest = entry.load_manifest(table.file_io()).await.unwrap();
            assert_eq!(manifest.entries().len(), 1);
            let me = &manifest.entries()[0];
            match entry.content {
                ManifestContentType::Data => {
                    assert_eq!(me.data_file().content_type(), DataContentType::Data);
                    assert_eq!(me.data_file().record_count(), 7);
                }
                ManifestContentType::Deletes => {
                    assert_eq!(
                        me.data_file().content_type(),
                        DataContentType::EqualityDeletes
                    );
                    assert_eq!(me.data_file().equality_ids(), Some(vec![1]));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_fast_append_file_with_incompatible_partition_value() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append();

        // check add data file with incompatible partition value
        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/3.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::string("test"))]))
            .build()
            .unwrap();

        let action = action.add_data_files(vec![data_file.clone()]);

        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_fast_append() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.fast_append();

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/3.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = action.add_data_files(vec![data_file.clone()]);
        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();
        let requirements = action_commit.take_requirements();

        // check updates and requirements
        assert!(
            matches!((&updates[0],&updates[1]), (TableUpdate::AddSnapshot { snapshot },TableUpdate::SetSnapshotRef { reference,ref_name }) if snapshot.snapshot_id() == reference.snapshot_id && ref_name == MAIN_BRANCH)
        );
        assert_eq!(
            vec![
                TableRequirement::UuidMatch {
                    uuid: table.metadata().uuid()
                },
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: MAIN_BRANCH.to_string(),
                    snapshot_id: table.metadata().current_snapshot_id
                }
            ],
            requirements
        );

        // check manifest list
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());
        assert_eq!(
            manifest_list.entries()[0].sequence_number,
            new_snapshot.sequence_number()
        );

        // check manifest
        let manifest = manifest_list.entries()[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(1, manifest.entries().len());
        assert_eq!(
            new_snapshot.sequence_number(),
            manifest.entries()[0]
                .sequence_number()
                .expect("Inherit sequence number by load manifest")
        );

        assert_eq!(
            new_snapshot.snapshot_id(),
            manifest.entries()[0].snapshot_id().unwrap()
        );
        assert_eq!(data_file, *manifest.entries()[0].data_file());
    }
}
