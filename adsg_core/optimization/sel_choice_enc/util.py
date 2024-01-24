"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import pickle
import itertools
import numpy as np
from typing import *
from adsg_core.optimization.sel_choice_enc.encoder import *
from adsg_core.optimization.sel_choice_enc.brute_force import *

__all__ = ['export_results', 'assert_behavior', 'validate_results', 'create_from_results']


def assert_behavior(encoder: SelectionChoiceEncoder):
    n_dv = encoder.n_choices
    n_opts = encoder.design_variable_options
    assert len(n_opts) == n_dv

    is_forced = encoder.design_variable_is_forced
    assert len(is_forced) == n_dv
    if n_dv > 0:
        not_forced = ~np.array(is_forced, dtype=bool)
        assert np.any(not_forced)  # At least one DV should be non-forced

        assert np.min(np.array(n_opts)[not_forced]) >= 2

    n_valid = encoder.n_valid
    assert 0 <= n_valid <= encoder.n_design_space
    if n_dv == 0:
        assert n_valid <= 1

    n_test_all = 1000
    x_all = status_all = None
    node_ids = encoder.node_ids
    if n_valid <= n_test_all:
        x_all, status_all = encoder.all_design_vectors_and_statuses
        assert x_all.shape == (n_valid, n_dv)
        assert status_all.shape == (n_valid, len(node_ids))

        for n in range(1, 4):
            for i_nodes in itertools.combinations(range(len(node_ids)), n):
                node_ids_query = [node_ids[i_n] for i_n in i_nodes]
                existence = encoder.get_node_existence_mask(node_ids_query)
                assert np.all(existence == (status_all[:, i_nodes] == Diag.CONFIRMED.value))

                for i_dv in np.random.randint(0, n_valid, 10):
                    existence_i = encoder.get_node_existence_mask(node_ids_query, i_dv=i_dv)
                    assert np.all(existence_i == (status_all[i_dv, i_nodes] == Diag.CONFIRMED.value))

    i_dv_test = np.arange(n_valid) if n_valid <= n_test_all else np.random.randint(0, n_valid, n_test_all)
    for i_dv in i_dv_test:
        i_exclude_all = set(range(n_valid))-{i_dv}
        dv, i_dv_ = encoder.get_design_vector_index([0]*n_dv, i_dv_excluded=i_exclude_all)
        assert i_dv_ == i_dv
        assert len(dv) == n_dv

        if x_all is not None:
            assert np.all(x_all[i_dv, :] == dv)
            node_exists = encoder.get_node_existence_mask(list(node_ids), i_dv=i_dv)
            assert np.all(node_exists == (status_all[i_dv, :] == Diag.CONFIRMED.value))

        dv_imp = np.array(dv, dtype=int)
        dv_imp[dv_imp == X_INACTIVE_VALUE] = 0
        dv_imp = list(dv_imp)

        dv_idx_set = encoder.get_design_vector_indices({ic: io for ic, io in enumerate(dv_imp)})
        assert dv_idx_set == {i_dv}

        dv_, i_dv_ = encoder.get_design_vector_index(dv_imp)
        assert i_dv_ == i_dv
        assert np.all(np.array(dv_) == np.array(dv))

        # Check that forced design variables are ignored
        for ix in np.where(is_forced)[0]:
            for value in range(n_opts[ix]):
                dv_mod = list(dv_imp)
                dv_mod[ix] = value

                dv_, i_dv_ = encoder.get_design_vector_index(dv_mod)
                assert i_dv_ == i_dv
                assert np.all(np.array(dv_) == np.array(dv))


def export_results(encoder: SelectionChoiceEncoder):
    x_all, status_all = encoder.all_design_vectors_and_statuses
    return {
        'kwargs': {
            'influence_matrix': encoder.influence_matrix,
            'node_ids': encoder.node_ids,
            'choice_opt_map': encoder.choice_opt_map,
            'choice_constraint_map': encoder.choice_constraint_map,
        },
        'dv_options': encoder.design_variable_options,
        'is_forced': encoder.design_variable_is_forced,
        'n_valid': encoder.n_valid,
        'x_all': x_all,
        'status_all': status_all,
    }


def create_from_results(encoder_factory: Callable[[Any], SelectionChoiceEncoder], results):
    if not isinstance(results, dict):
        with open(results, 'rb') as fp:
            results = pickle.load(fp)

    return encoder_factory(**results['kwargs']), results


def validate_results(encoder_factory: Callable[[Any], SelectionChoiceEncoder], results):
    encoder, results = create_from_results(encoder_factory, results)

    assert np.all(np.array(encoder.design_variable_options) == np.array(results['dv_options']))
    assert np.all(np.array(encoder.design_variable_is_forced) == np.array(results['is_forced']))
    assert encoder.n_valid == results['n_valid']

    x_ref: np.ndarray = results['x_all']
    x_ref_map = {tuple(xi): ix for ix, xi in enumerate(x_ref)}
    status_ref: np.ndarray = results['status_all']

    # Ignore actual order
    x_all, status_all = encoder.all_design_vectors_and_statuses
    assert x_all.shape == x_ref.shape
    i_ref = np.zeros((x_ref.shape[0],), dtype=int)
    for ix, xi in enumerate(x_all):
        ix_ref = x_ref_map.get(tuple(xi))
        assert ix_ref is not None
        i_ref[ix] = ix_ref
    assert len(np.unique(i_ref)) == len(i_ref)

    x_ref = x_ref[i_ref, :]
    status_ref = status_ref[i_ref, :]
    assert np.all(x_all == x_ref)
    assert np.all(status_all == status_ref)

    for i_dv in range(x_all.shape[0]):
        dv = x_all[i_dv, :]
        dv_imp = dv.copy()
        dv_imp[dv_imp == X_INACTIVE_VALUE] = 0

        dv_, i_dv_ = encoder.get_design_vector_index(list(dv_imp))
        assert i_dv_ == i_dv
        assert np.all(np.array(dv_) == dv)

    n_opts = encoder.design_variable_options
    for n in range(1, 4):
        for i_x in itertools.combinations(range(x_all.shape[1]), n):
            for i_x_values in itertools.product(*[list(range(n_opts[i])) for i in i_x]):
                choice_opt_idx_map = {i_choice: i_x_values[i] for i, i_choice in enumerate(i_x)}
                i_dv_matched = BruteForceSelectionChoiceEncoder._match_dv_indices(x_all, choice_opt_idx_map)
                assert i_dv_matched == encoder.get_design_vector_indices(choice_opt_idx_map)

    node_ids = encoder.node_ids
    for n in range(1, 4):
        for i_nodes in itertools.combinations(range(len(node_ids)), n):
            node_ids_query = [node_ids[i_n] for i_n in i_nodes]
            existence = encoder.get_node_existence_mask(node_ids_query)
            assert np.all(existence == (status_all[:, i_nodes] == Diag.CONFIRMED.value))

            for i_dv in range(status_all.shape[0]):
                existence_i = encoder.get_node_existence_mask(node_ids_query, i_dv=i_dv)
                assert np.all(existence_i == (status_all[i_dv, i_nodes] == Diag.CONFIRMED.value))

    assert_behavior(encoder)
