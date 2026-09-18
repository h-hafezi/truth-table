use std::collections::HashSet;

use ark_ff::{Field as _, PrimeField, Zero};
use ark_piop::{
    DefaultSnarkBackend, SnarkBackend, arithmetic::mat_poly::mle::MLE,
    test_utils::prelude_with_vars,
};
use indexmap::IndexMap;

use super::{
    GadgetNode, add_match_count_claims_prover, add_match_count_claims_verifier,
    ensure_match_count_no_wrap, union_activator_bool_table_prover,
};
use crate::irs::nodes::IsNode;

type B = DefaultSnarkBackend;
type F = <B as SnarkBackend>::F;

#[test]
fn union_activator_booleanity_is_ungated() {
    let (mut prover, _) = prelude_with_vars::<B>(2).expect("SRS setup");
    let half = F::from(2u64).inverse().expect("two is invertible");
    let activator = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(
            2,
            vec![half, half, F::from(1u64), F::zero()],
        ))
        .expect("commit union activator");
    let mut union_polys = IndexMap::new();
    union_polys.insert(arithmetic::ACTIVATOR_FIELD.clone(), activator);
    let union = arithmetic::table::TrackedTable::new(
        Some(datafusion::arrow::datatypes::Schema::new(vec![
            arithmetic::ACTIVATOR_FIELD.as_ref().clone(),
        ])),
        union_polys,
        2,
    );

    let bool_input = union_activator_bool_table_prover(&union);
    assert!(
        bool_input.activator_tracked_poly().is_none(),
        "the value being checked must not also gate its own BoolCheck"
    );
    let data_indices = bool_input.data_tracked_polys_indices();
    assert_eq!(data_indices.len(), 1);
    assert_eq!(
        bool_input
            .tracked_col_by_ind(data_indices[0])
            .data_tracked_poly()
            .evaluations(),
        vec![half, half, F::from(1u64), F::zero()]
    );

    let child_names = GadgetNode::<B>::new()
        .children()
        .into_iter()
        .map(|child| child.name())
        .collect::<HashSet<_>>();
    assert!(
        child_names.contains("Bool"),
        "MatchPair must execute the union-activator BoolCheck child"
    );
}

#[test]
fn output_activator_sum_is_bound_to_pair_count() {
    let (mut prover, mut verifier) = prelude_with_vars::<B>(4).expect("SRS setup");

    // The pair-count polynomial sums to two, while the claimed join output
    // contains only one active row. This models an omitted valid pair.
    let pair_count_evals = vec![F::from(1u64), F::from(1u64), F::zero(), F::zero()];
    let output_active_evals = vec![F::from(1u64), F::zero(), F::zero(), F::zero()];
    let pair_count = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(2, pair_count_evals))
        .expect("commit pair-count polynomial");
    let output_active = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(2, output_active_evals))
        .expect("commit output activator");
    let claimed_count = F::from(2u64);
    let result =
        add_match_count_claims_prover(&mut prover, &pair_count, &output_active, claimed_count);
    if cfg!(feature = "honest-prover") {
        assert!(
            result.is_err(),
            "honest-prover must reject the false count claim at registration"
        );
        return;
    }
    result.expect("register transcript-bound match count");
    let proof = prover
        .build_proof()
        .expect("a false Sumcheck claim may still be serialized");
    verifier.set_proof(proof);
    let pair_count_oracle = verifier
        .track_mv_com_by_id(pair_count.id())
        .expect("track pair-count commitment");
    let output_active_oracle = verifier
        .track_mv_com_by_id(output_active.id())
        .expect("track output-activator commitment");
    add_match_count_claims_verifier(&mut verifier, &pair_count_oracle, &output_active_oracle)
        .expect("mirror transcript-bound match count");
    assert!(
        verifier.verify().is_err(),
        "MatchPair must reject an output activator with the wrong active-row count"
    );
}

#[test]
fn match_count_comparison_handles_different_domains() {
    let (mut prover, mut verifier) = prelude_with_vars::<B>(4).expect("SRS setup");
    let pair_count = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(
            2,
            vec![F::from(1u64), F::from(1u64), F::zero(), F::zero()],
        ))
        .expect("commit pair-count polynomial");
    let output_active = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(
            1,
            vec![F::from(1u64), F::zero()],
        ))
        .expect("commit output activator");
    let result =
        add_match_count_claims_prover(&mut prover, &pair_count, &output_active, F::from(2u64));
    if cfg!(feature = "honest-prover") {
        assert!(
            result.is_err(),
            "honest-prover must reject the false cross-domain count at registration"
        );
        return;
    }
    result.expect("register cross-domain match count");
    let proof = prover
        .build_proof()
        .expect("a false cross-domain claim may still be serialized");
    verifier.set_proof(proof);
    let pair_count_oracle = verifier
        .track_mv_com_by_id(pair_count.id())
        .expect("track pair-count commitment");
    let output_active_oracle = verifier
        .track_mv_com_by_id(output_active.id())
        .expect("track output-activator commitment");
    add_match_count_claims_verifier(&mut verifier, &pair_count_oracle, &output_active_oracle)
        .expect("mirror cross-domain match count");
    assert!(
        verifier.verify().is_err(),
        "domain normalization must preserve counts rather than averages"
    );
}

#[test]
fn shared_match_count_accepts_equal_sums() {
    let (mut prover, mut verifier) = prelude_with_vars::<B>(4).expect("SRS setup");
    let pair_count = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(
            2,
            vec![F::from(1u64), F::from(1u64), F::zero(), F::zero()],
        ))
        .expect("commit pair-count polynomial");
    let output_active = prover
        .track_and_commit_mat_mv_poly(&MLE::from_evaluations_vec(
            1,
            vec![F::from(1u64), F::from(1u64)],
        ))
        .expect("commit output activator");
    let claimed_count = F::from(2u64);
    add_match_count_claims_prover(&mut prover, &pair_count, &output_active, claimed_count)
        .expect("register shared match count");
    let proof = prover.build_proof().expect("build matching proof");
    assert!(
        proof.miscellaneous_field_elements.is_empty(),
        "the match count must not use an unbound miscellaneous proof field"
    );
    assert!(
        proof
            .mv_pcs_subproof
            .unique_constants
            .values()
            .any(|value| *value == claimed_count),
        "the shared count must be a transcript-bound proof constant"
    );
    verifier.set_proof(proof);
    let pair_count_oracle = verifier
        .track_mv_com_by_id(pair_count.id())
        .expect("track pair-count commitment");
    let output_active_oracle = verifier
        .track_mv_com_by_id(output_active.id())
        .expect("track output-activator commitment");
    add_match_count_claims_verifier(&mut verifier, &pair_count_oracle, &output_active_oracle)
        .expect("mirror shared match count");
    verifier.verify().expect("equal counts must verify");
}

#[test]
fn match_count_capacity_must_not_wrap_the_field() {
    let modulus_bits = F::MODULUS_BIT_SIZE as usize;
    assert!(ensure_match_count_no_wrap::<F>(1, 1, 1).is_ok());
    assert!(ensure_match_count_no_wrap::<F>(modulus_bits - 1, 1, 1).is_err());
    assert!(ensure_match_count_no_wrap::<F>(1, 1, modulus_bits).is_err());
    assert!(ensure_match_count_no_wrap::<F>(usize::MAX, 1, 1).is_err());
}
