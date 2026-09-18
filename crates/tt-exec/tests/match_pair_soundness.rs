//! End-to-end regression for MatchPair's additional Boolean/count claims.

use std::{fs::File, path::Path, sync::Arc};

use anyhow::Result;
use arithmetic::{ACTIVATOR_COL_NAME, ROW_ID_COL_NAME};
use datafusion::arrow::{
    array::{ArrayRef, BooleanArray, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use tt_exec::{
    commit::CommitBuilder, prove::ProveBuilder, setup::SetupBuilder, verify::VerifyBuilder,
};

fn write_batch(path: &Path, batch: &RecordBatch) -> Result<()> {
    let mut writer = ArrowWriter::try_new(File::create(path)?, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

fn active_table(
    key_name: &str,
    value_name: &str,
    keys: Vec<i64>,
    values: Vec<i64>,
) -> Result<RecordBatch> {
    let rows = keys.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new(key_name, DataType::Int64, false),
        Field::new(value_name, DataType::Int64, false),
        Field::new(ROW_ID_COL_NAME, DataType::Int64, false),
        Field::new(ACTIVATOR_COL_NAME, DataType::Boolean, false),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(values)) as ArrayRef,
            Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef,
            Arc::new(BooleanArray::from(vec![true; rows])) as ArrayRef,
        ],
    )?)
}

fn parquet_row_count(path: &Path) -> Result<usize> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    Ok(reader
        .collect::<std::result::Result<Vec<_>, _>>()?
        .iter()
        .map(RecordBatch::num_rows)
        .sum())
}

#[tokio::test]
async fn honest_duplicate_key_join_still_proves_and_verifies() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let left_path = dir.path().join("match_left.parquet");
    let right_path = dir.path().join("match_right.parquet");
    let pk_path = dir.path().join("match.pk");
    let vk_path = dir.path().join("match.vk");

    // Key 1 occurs twice on the left and once on the right, so the honest join
    // has two output pairs.  This exercises multiplicities as well as the new
    // union-activator BoolCheck and shared output-count Sumcheck target.
    write_batch(
        &left_path,
        &active_table("l_key", "l_value", vec![1, 1, 2], vec![10, 11, 20])?,
    )?;
    write_batch(
        &right_path,
        &active_table("r_key", "r_value", vec![1, 3], vec![100, 300])?,
    )?;

    SetupBuilder::new()
        .with_size_label(Some("16".to_owned()))
        .with_pk_path(Some(pk_path.clone()))
        .with_vk_path(Some(vk_path.clone()))
        .build()?
        .run()?;
    let left_oracle = CommitBuilder::new()
        .with_parquet_path(left_path.clone())
        .with_pk_path(pk_path.clone())
        .with_output_path(Some(dir.path().join("match_left.oracle")))
        .build()?
        .run()
        .await?;
    let right_oracle = CommitBuilder::new()
        .with_parquet_path(right_path.clone())
        .with_pk_path(pk_path.clone())
        .with_output_path(Some(dir.path().join("match_right.oracle")))
        .build()?
        .run()
        .await?;

    let query = "SELECT l.l_value, r.r_value \
        FROM match_left l JOIN match_right r ON l.l_key = r.r_key";
    let output = ProveBuilder::new()
        .with_query(query.to_owned())
        .with_parquet_paths(vec![left_path, right_path])
        .with_oracle_paths(vec![left_oracle.clone(), right_oracle.clone()])
        .with_pk_path(pk_path)
        .with_output_path(Some(dir.path().join("match.proof")))
        .build()?
        .run()
        .await?;
    assert_eq!(parquet_row_count(&output.result_path)?, 2);

    VerifyBuilder::new()
        .with_query(query.to_owned())
        .with_oracle_paths(vec![left_oracle, right_oracle])
        .with_proof_path(output.proof_path)
        .with_result_path(output.result_path)
        .with_vk_path(vk_path)
        .build()?
        .run()
        .await
}
