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

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::Schema;
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// A transaction action that replaces the table's current schema with a new
/// target schema.
///
/// Callers compute the desired schema (e.g. by adding/dropping columns) and
/// hand it to [`Self::set_schema`]. On commit, this emits two table updates:
///
/// - [`TableUpdate::AddSchema`] with the target schema.
/// - [`TableUpdate::SetCurrentSchema`] with `schema_id = -1` (the "last added
///   schema" sentinel), so we don't have to allocate a non-clashing schema id
///   ourselves — `TableMetadataBuilder` does that.
///
/// Three requirements are attached so concurrent schema/column changes fail
/// loudly rather than silently overwrite each other:
///
/// - `UuidMatch` — guards against the table being recreated under us.
/// - `CurrentSchemaIdMatch` — guards against a concurrent schema swap.
/// - `LastAssignedFieldIdMatch` — guards against a concurrent column add.
///
/// Note: this action does not enforce schema-evolution compatibility rules
/// (e.g. "you may add columns but not remove them"). The metadata builder
/// also doesn't — it's the caller's responsibility to construct a schema
/// that's a valid evolution of the current one.
#[derive(Debug, Default)]
pub struct UpdateSchemaAction {
    target_schema: Option<Schema>,
}

impl UpdateSchemaAction {
    /// Creates a new [`UpdateSchemaAction`] with no schema set.
    pub fn new() -> Self {
        UpdateSchemaAction {
            target_schema: None,
        }
    }

    /// Sets the target schema this action will commit.
    pub fn set_schema(mut self, schema: Schema) -> Self {
        self.target_schema = Some(schema);
        self
    }
}

#[async_trait]
impl TransactionAction for UpdateSchemaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let target = self.target_schema.clone().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Schema is not set for UpdateSchemaAction!",
            )
        })?;

        let metadata = table.metadata();

        let updates = vec![
            TableUpdate::AddSchema {
                schema: target.clone(),
            },
            // `-1` is the LAST_ADDED sentinel honored by
            // `TableMetadataBuilder::set_current_schema`.
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];

        let requirements = vec![
            TableRequirement::UuidMatch {
                uuid: metadata.uuid(),
            },
            TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: metadata.current_schema_id(),
            },
            TableRequirement::LastAssignedFieldIdMatch {
                last_assigned_field_id: metadata.last_column_id(),
            },
        ];

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use as_any::Downcast;

    use super::UpdateSchemaAction;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};
    use crate::transaction::Transaction;
    use crate::transaction::action::{ApplyTransactionAction, TransactionAction};
    use crate::transaction::tests::make_v2_table;

    fn schema_with_one_int_column() -> Schema {
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn applies_action_to_transaction() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let tx = UpdateSchemaAction::new()
            .set_schema(schema_with_one_int_column())
            .apply(tx)
            .unwrap();

        assert_eq!(tx.actions.len(), 1);

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateSchemaAction>()
            .expect("UpdateSchemaAction was not applied to Transaction!");

        assert!(action.target_schema.is_some());
    }

    #[tokio::test]
    async fn commit_without_schema_set_errors() {
        let table = make_v2_table();
        let action = std::sync::Arc::new(UpdateSchemaAction::new());
        let result = action.commit(&table).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.message().contains("Schema is not set"));
    }

    #[tokio::test]
    async fn commit_emits_add_schema_and_set_current_schema_updates() {
        let table = make_v2_table();
        let action = std::sync::Arc::new(
            UpdateSchemaAction::new().set_schema(schema_with_one_int_column()),
        );
        let mut commit = action.commit(&table).await.unwrap();
        let updates = commit.take_updates();
        let requirements = commit.take_requirements();

        assert_eq!(updates.len(), 2);
        assert!(matches!(
            updates[0],
            crate::TableUpdate::AddSchema { .. }
        ));
        assert!(matches!(
            updates[1],
            crate::TableUpdate::SetCurrentSchema { schema_id: -1 }
        ));

        // Three requirements, in a fixed order.
        assert_eq!(requirements.len(), 3);
        assert!(matches!(
            requirements[0],
            crate::TableRequirement::UuidMatch { .. }
        ));
        assert!(matches!(
            requirements[1],
            crate::TableRequirement::CurrentSchemaIdMatch { .. }
        ));
        assert!(matches!(
            requirements[2],
            crate::TableRequirement::LastAssignedFieldIdMatch { .. }
        ));
    }
}
