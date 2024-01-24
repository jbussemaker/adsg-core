import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.time_limiter import *
from adsg_core.optimization.assign_enc.lazy_encoding import *
from adsg_core.optimization.assign_enc.encoder_registry import *
from adsg_core.optimization.assign_enc.patterns.encoder import *
from adsg_core.optimization.assign_enc.assignment_manager import *
from adsg_core.optimization.assign_enc.cache import get_cache_path

__all__ = ['EncoderSelector']

log = logging.getLogger('adsg_core.selector')


class EncoderSelector:
    """Algorithm for automatically selecting the best encoder for a given assignment problem."""

    _global_disable_cache = False  # For testing/experiments
    _exclude_pattern_encoders = False
    _print_stats = False
    _numba_initialized = False

    encoding_timeout = .25  # sec
    n_mat_max_eager = 1e3
    imputation_ratio_limits = [10, 40, 100]

    limit_dist_corr_time = True
    min_distance_correlation = .7

    def __init__(self, settings: MatrixGenSettings):
        self.settings: MatrixGenSettings = settings

        self.lazy_imputer = DEFAULT_LAZY_IMPUTER
        self.eager_imputer = DEFAULT_EAGER_IMPUTER
        self._last_selection_stage = None

    def reset_cache(self):
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_best_assignment_manager(self, cache=True, limit_time=True) -> AssignmentManagerBase:
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if not self._global_disable_cache and cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)
        self.initialize_numba()

        enc_timeout, n_mme, limit_dc_time = self.encoding_timeout, self.n_mat_max_eager, self.limit_dist_corr_time
        if not limit_time:
            self.encoding_timeout = 10
            self.n_mat_max_eager = 1e5
            # self.limit_dist_corr_time = False

        assignment_manager = self._get_best_assignment_manager()

        if not limit_time:
            self.encoding_timeout, self.n_mat_max_eager, self.limit_dist_corr_time = enc_timeout, n_mme, limit_dc_time

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(assignment_manager, fp)
        return assignment_manager

    def initialize_numba(self):
        """Run a simple encoding problem so that numba functions are JIT compiled and compilation time doesn't influence
        selection timeouts"""
        if self.__class__._numba_initialized:
            return
        self.__class__._numba_initialized = True

        enc_timeout, settings = self.encoding_timeout, self.settings

        self.encoding_timeout = 10
        self.settings = MatrixGenSettings(
            src=[Node([0, 1]) for _ in range(2)], tgt=[Node([0, 1]) for _ in range(2)],
        )
        log.debug('Initializing numba...')
        try:
            self._exclude_pattern_encoders = True
            self._get_best_assignment_manager()
        finally:
            self.encoding_timeout, self.settings = enc_timeout, settings
            self._exclude_pattern_encoders = False

    def _get_best_assignment_manager(self) -> AssignmentManager:
        log.debug('Counting matrices...')
        self._provision_agg_matrix_cache()
        n_mat, n_exist = self._get_n_mat()
        log.debug(f'{n_mat} matrices ({n_exist} existence scheme{"s" if n_exist != 1 else ""})')

        dist_corr_lazy_limit = self.imputation_ratio_limits[0]
        dist_corr_eager_limit = self.imputation_ratio_limits[-1]
        high_imp_ratio_detect_limit = self.imputation_ratio_limits[-1]

        def _instantiate_manager(encoder):
            cls = LazyAssignmentManager if isinstance(encoder, LazyEncoder) else AssignmentManager
            return cls(self.settings, encoder)

        def _create_managers(encoders, imputer_factory, dist_corr_limit, limit_dist_corr_time=None):
            assignment_mgr = []
            scoring = {'n_des_pts': [], 'imp_ratio': [], 'inf_idx': [], 'dist_corr': []}
            with Encoder.with_early_detect_high_imp_ratio(high_imp_ratio=high_imp_ratio_detect_limit):
                for encoder_factory in encoders:
                    # Create encoder
                    encoder = encoder_factory(imputer_factory())

                    # Create assignment manager
                    log.debug(f'Encoding {encoder!s}')
                    try:
                        assignment_manager = run_timeout(self.encoding_timeout, _instantiate_manager, encoder)

                    except InvalidPatternEncoder:
                        continue

                    except DetectedHighImpRatio:
                        log.debug('Excessively high imputation ratio detected!')
                        continue

                    except (TimeoutError, MemoryError):
                        log.debug('Encoding timeout!')
                        continue

                    # Get metrics
                    n_design_points = assignment_manager.encoder.get_n_design_points()
                    imputation_ratio = self._get_imp_ratio(
                        n_design_points, n_mat, n_exist, assignment_manager=assignment_manager)
                    information_index = assignment_manager.encoder.get_information_index()

                    distance_correlation = np.nan
                    if imputation_ratio <= dist_corr_limit:
                        if limit_dist_corr_time:
                            limit_dist_corr_time = self.limit_dist_corr_time
                        if limit_dist_corr_time:
                            try:
                                distance_correlation = run_timeout(
                                    self.encoding_timeout, self._get_dist_corr, assignment_manager)
                            except (TimeoutError, MemoryError):
                                pass
                        else:
                            distance_correlation = self._get_dist_corr(assignment_manager)

                    assignment_mgr.append(assignment_manager)
                    scoring['n_des_pts'].append(n_design_points)
                    scoring['imp_ratio'].append(imputation_ratio)
                    scoring['inf_idx'].append(information_index)
                    scoring['dist_corr'].append(distance_correlation)

                    # Stop if we found a perfect encoder
                    if imputation_ratio == 1. and distance_correlation >= .95:
                        break

            # Equalize distance correlations
            df = pd.DataFrame(data=scoring)
            unique_values, unique_indices_list = np.unique(df[['imp_ratio', 'inf_idx']].values, axis=0, return_inverse=True)
            dist_corr_values = df.dist_corr.values
            for i_unique in range(len(unique_values)):
                unique_indices, = np.where(unique_indices_list == i_unique)
                if len(unique_indices) > 1:
                    values = df.dist_corr.values[unique_indices]
                    values = values[~np.isnan(values)]
                    if len(values) > 0:
                        dist_corr_values[unique_indices] = np.mean(values)
            df.dist_corr = pd.Series(index=df.dist_corr.index, data=np.round(dist_corr_values/.01)*.01)

            return df, assignment_mgr

        # Special case if there are no possible connections
        if n_mat == 0:
            return _instantiate_manager(DEFAULT_EAGER_ENCODER())

        def _print_stats(i_select):
            if not self._print_stats:
                return
            df_score['name'] = [str(am.encoder) for am in assignment_managers]
            df_score['selection_stage'] = [self._last_selection_stage if i_am == i_select else ''
                                           for i_am in range(len(df_score))]
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.expand_frame_repr', False)
            print(df_score)

        # Try pattern encoders
        df_score, assignment_managers = None, []
        if not self._exclude_pattern_encoders:
            df_score, assignment_managers = _create_managers(
                PATTERN_ENCODERS, self.lazy_imputer, dist_corr_lazy_limit, limit_dist_corr_time=False)
            i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, n_priority=5)
            if i_best is not None:
                self._last_selection_stage = '0_pattern'
                _print_stats(i_best)
                return assignment_managers[i_best]

        # If there are not too many matrices, encode all encoders except enumeration-based
        initially_all = False
        if n_mat is not None and n_mat <= self.n_mat_max_eager:
            df_eager, eager_assignment_managers = \
                _create_managers(EAGER_ENCODERS, self.eager_imputer, dist_corr_eager_limit)
            df_score = pd.concat([df_score, df_eager], ignore_index=True) if df_score is not None else df_eager

            df_lazy, lazy_assignment_mgr = _create_managers(LAZY_ENCODERS, self.lazy_imputer, dist_corr_lazy_limit)
            df_score = pd.concat([df_score, df_lazy], ignore_index=True)
            assignment_managers += eager_assignment_managers+lazy_assignment_mgr
            initially_all = True

        # Otherwise initially only try lazy encoders
        else:
            df_lazy, lazy_assignment_managers = _create_managers(LAZY_ENCODERS, self.lazy_imputer, dist_corr_lazy_limit)
            df_score = pd.concat([df_score, df_lazy], ignore_index=True) if df_score is not None else df_lazy
            assignment_managers += lazy_assignment_managers

        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, n_priority=4)
        if i_best is not None:
            self._last_selection_stage = '1_init_all' if initially_all else '1_init_lazy'
            _print_stats(i_best)
            return assignment_managers[i_best]

        # Select based on information index (higher information indices correlate with higher distance correlation for
        # non-enumeration-based encoders)
        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, n_priority=4, by_inf_idx=True)
        if i_best is not None:
            self._last_selection_stage = '2_init_inf_idx'
            _print_stats(i_best)
            return assignment_managers[i_best]

        # If none found, add the other type and remaining priority areas
        if not initially_all:
            df_other, other_assignment_mgr = _create_managers(EAGER_ENCODERS, self.eager_imputer, dist_corr_eager_limit)
            df_score = pd.concat([df_other, df_score], ignore_index=True)
            assignment_managers = other_assignment_mgr+assignment_managers

            i_best = self._get_best(df_score, knows_n_mat=n_mat is not None)
            if i_best is not None:
                self._last_selection_stage = '3_all'
                _print_stats(i_best)
                return assignment_managers[i_best]

        # Also include enumeration-based encoders
        df_enum, enum_assignment_mgr = _create_managers(EAGER_ENUM_ENCODERS, self.lazy_imputer, dist_corr_eager_limit)
        if np.all(np.isnan(df_score.dist_corr)):
            df_enum.dist_corr = np.nan
        df_score = pd.concat([df_score, df_enum], ignore_index=True)
        assignment_managers = assignment_managers+enum_assignment_mgr

        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None)
        if i_best is not None:
            self._last_selection_stage = '4_all_enum'
            _print_stats(i_best)
            return assignment_managers[i_best]

        # Select based on information index
        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, by_inf_idx=True)
        if i_best is None:
            raise RuntimeError(f'Cannot find best encoder, try increasing timeout')
        self._last_selection_stage = '4_all_inf_idx'
        _print_stats(i_best)
        return assignment_managers[i_best]

    def _get_imp_ratio(self, n_design_points: int, n_mat: int = None, n_exist: int = None,
                       assignment_manager: AssignmentManagerBase = None) -> float:
        if assignment_manager is not None:
            return assignment_manager.encoder.get_imputation_ratio(per_existence=True)
        return ((n_design_points*n_exist)/n_mat) if n_mat is not None else n_design_points

    def _get_dist_corr(self, assignment_manager: AssignmentManagerBase) -> float:
        return assignment_manager.encoder.get_distance_correlation(minimum=True)

    def reset_agg_matrix_cache(self):
        self._get_matrix_gen().reset_agg_matrix_cache()

    def _provision_agg_matrix_cache(self):
        log.debug('Generating aggregate matrix for eager encoders...')
        self._get_matrix_gen().get_agg_matrix(cache=True)

    def _get_best(self, df_scores: pd.DataFrame, knows_n_mat, n_priority: int = None,
                  by_inf_idx=False) -> Optional[int]:
        if len(df_scores) == 0:
            return
        df_scores = df_scores.copy()
        df_scores['idx'] = df_scores.index

        corr_col = 'inf_idx' if by_inf_idx else 'dist_corr'

        # If we don't accurately know the total amount of matrices, normalize to the lowest imputation ratio
        if not knows_n_mat:
            df_scores.imp_ratio /= df_scores.imp_ratio.min()

        def _return_best_within_priority_area(df_):
            if by_inf_idx:
                # Get rows with the minimum imputation ratio
                min_imp_ratio = np.min(df_.imp_ratio)
                min_imp_ratio_mask = df_.imp_ratio == min_imp_ratio

                # Within these rows, get the row with the maximum information index
                i_max_inf_idx = np.argmax(df_.inf_idx[min_imp_ratio_mask])
                min_imp_idx = df_.idx.values[min_imp_ratio_mask]
                return int(min_imp_idx[i_max_inf_idx])

            # Get rows with maximum distance correlation
            max_dist_corr = np.nanmax(df_.dist_corr)
            if np.isnan(max_dist_corr):
                return
            max_dist_corr_mask = df_.dist_corr == max_dist_corr

            # Within these rows, get the row with the minimum imputation ratio
            i_min_imp_rat = np.nanargmin(df_.imp_ratio[max_dist_corr_mask])
            min_imp_idx = df_.idx.values[max_dist_corr_mask]
            return int(min_imp_idx[i_min_imp_rat])

        # Check for points with low imputation ratio first
        df_imp_bands = [df_scores[df_scores.imp_ratio == 1]]
        df_imp_bands += [df_scores[df_scores.imp_ratio <= ir_limit] for ir_limit in self.imputation_ratio_limits]
        df_imp_bands += [df_scores]

        if by_inf_idx:
            df_priority = [
                df_imp_bands[0][df_imp_bands[0][corr_col] >= self.min_distance_correlation],
                df_imp_bands[1][df_imp_bands[1][corr_col] >= self.min_distance_correlation],
                df_imp_bands[0][df_imp_bands[0][corr_col] >= .5 * self.min_distance_correlation],
                df_imp_bands[1][df_imp_bands[1][corr_col] >= .5 * self.min_distance_correlation],
                df_imp_bands[0][df_imp_bands[0][corr_col] > 0],
                df_imp_bands[1][df_imp_bands[1][corr_col] > 0],
                df_imp_bands[0],
                df_imp_bands[1],
            ]
        else:
            df_priority = [
                df_imp_bands[0][df_imp_bands[0][corr_col] >= self.min_distance_correlation],
                df_imp_bands[1][df_imp_bands[1][corr_col] >= self.min_distance_correlation],
                df_imp_bands[0][df_imp_bands[0][corr_col] >= .5 * self.min_distance_correlation],
                df_imp_bands[1][df_imp_bands[1][corr_col] >= .5 * self.min_distance_correlation],
                # df_imp_bands[1][(df_imp_bands[1][corr_col] >= 0) & (df_imp_bands[1].inf_idx > 0)],
                df_imp_bands[1][df_imp_bands[1][corr_col] >= 0],
            ]
        for i in range(2, len(df_imp_bands)):
            df_priority += [
                df_imp_bands[i][df_imp_bands[i][corr_col] >= self.min_distance_correlation],
                df_imp_bands[i][df_imp_bands[i][corr_col] >= .5 * self.min_distance_correlation],
                df_imp_bands[i][df_imp_bands[i][corr_col] > 0],
                df_imp_bands[i],
            ]

        for i, df in enumerate(df_priority):
            if n_priority is not None and i >= n_priority:
                return
            if len(df) > 0:
                return _return_best_within_priority_area(df)
        raise RuntimeError('No encoders available!')

    def _get_n_mat(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            matrix_gen = self._get_matrix_gen()
            n_mat_total = run_timeout(
                self.encoding_timeout, lambda: matrix_gen.count_all_matrices(max_by_existence=False))
            n_existence = len(list(matrix_gen.iter_existence()))
            return n_mat_total, n_existence
        except TimeoutError:
            return None, None

    def _get_matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return AggregateAssignmentMatrixGenerator(self.settings)

    def _get_cache_key(self):
        return self.settings.get_cache_key()

    def _cache_path(self, sub_path=None):
        sel_cache_folder = 'encoder_cache'
        sub_path = os.path.join(sel_cache_folder, sub_path) if sub_path is not None else sel_cache_folder
        return get_cache_path(sub_path=sub_path)
