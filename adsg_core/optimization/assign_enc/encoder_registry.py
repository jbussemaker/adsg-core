from adsg_core.optimization.assign_enc.patterns import *
from adsg_core.optimization.assign_enc.enumerating import *
from adsg_core.optimization.assign_enc.lazy.encodings import *
from adsg_core.optimization.assign_enc.lazy.imputation import *
from adsg_core.optimization.assign_enc.eager.encodings import *
from adsg_core.optimization.assign_enc.eager.imputation import *


EAGER_ENCODERS = [
    lambda imp: DirectMatrixEncoder(imp),
    # lambda imp: DirectMatrixEncoder(imp, remove_gaps=False),
    lambda imp: ElementGroupedEncoder(imp),
    # lambda imp: ElementGroupedEncoder(imp, normalize_within_group=False),
    lambda imp: ConnIdxGroupedEncoder(imp),
    lambda imp: ConnIdxGroupedEncoder(imp, by_src=False),
    lambda imp: ConnIdxGroupedEncoder(imp, binary=True),
    lambda imp: ConnIdxGroupedEncoder(imp, by_src=False, binary=True),

    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), OneVarLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), OneVarLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), OneVarLocationGrouper()),

    # lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), FlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), FlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), FlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), FlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), FlatIndexLocationGrouper()),

    # lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelFlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), RelFlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), RelFlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), RelFlatIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), RelFlatIndexLocationGrouper()),

    # lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), CoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), CoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), CoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), CoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), CoordIndexLocationGrouper()),

    # lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelCoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), RelCoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), RelCoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), RelCoordIndexLocationGrouper()),
    # lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), RelCoordIndexLocationGrouper()),
]

DEFAULT_EAGER_ENCODER = lambda: DirectMatrixEncoder(DEFAULT_EAGER_IMPUTER())

EAGER_IMPUTERS = [
    lambda: FirstImputer(),
    lambda: AutoModImputer(),
    lambda: AutoModImputer(reverse=True),
    lambda: DeltaImputer(),
    lambda: ClosestImputer(),
    lambda: ClosestImputer(euclidean=False),
    lambda: ConstraintViolationImputer(),
]

DEFAULT_EAGER_IMPUTER = AutoModImputer

LAZY_ENCODERS = [
    lambda imp: LazyDirectMatrixEncoder(imp),

    lambda imp: LazyAmountFirstEncoder(imp, FlatLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    # lambda imp: LazyAmountFirstEncoder(imp, TotalLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    # lambda imp: LazyAmountFirstEncoder(imp, SourceLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    lambda imp: LazyAmountFirstEncoder(imp, SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder()),

    lambda imp: LazyConnIdxMatrixEncoder(imp, FlatConnCombsEncoder()),
    lambda imp: LazyConnIdxMatrixEncoder(imp, FlatConnCombsEncoder(), by_src=False),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, FlatConnCombsEncoder(), amount_first=True),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, FlatConnCombsEncoder(), by_src=False, amount_first=True),
    lambda imp: LazyConnIdxMatrixEncoder(imp, GroupedConnCombsEncoder()),
    lambda imp: LazyConnIdxMatrixEncoder(imp, GroupedConnCombsEncoder(), by_src=False),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, GroupedConnCombsEncoder(), amount_first=True),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, GroupedConnCombsEncoder(), by_src=False, amount_first=True),
    lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder()),
    lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(), by_src=False),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(), amount_first=True),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(), by_src=False, amount_first=True),
    lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(binary=True)),
    lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(binary=True), by_src=False),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(binary=True), amount_first=True),
    # lambda imp: LazyConnIdxMatrixEncoder(imp, ConnIdxCombsEncoder(binary=True), by_src=False, amount_first=True),
]

DEFAULT_LAZY_ENCODER = lambda: LazyDirectMatrixEncoder(DEFAULT_LAZY_IMPUTER())

EAGER_ENUM_ENCODERS = [  # Encoders only based on counting the possibilities, ignoring the actual connection patterns
    lambda imp: EnumOrdinalEncoder(imp),
    lambda imp: EnumRecursiveEncoder(imp, n_divide=2),
    lambda imp: EnumRecursiveEncoder(imp, n_divide=3),
    lambda imp: EnumRecursiveEncoder(imp, n_divide=4),
]

PATTERN_ENCODERS = [  # Encoders for specific assignment patterns
    lambda imp: CombiningPatternEncoder(imp),
    lambda imp: AssigningPatternEncoder(imp),
    lambda imp: PartitioningPatternEncoder(imp),
    lambda imp: ConnectingPatternEncoder(imp),
    lambda imp: PermutingPatternEncoder(imp),
    lambda imp: UnorderedCombiningPatternEncoder(imp),
]

LAZY_IMPUTERS = [
    lambda: LazyFirstImputer(),
    lambda: LazyDeltaImputer(),
    lambda: LazyClosestImputer(),
    lambda: LazyClosestImputer(euclidean=False),
    lambda: LazyConstraintViolationImputer(),
]

DEFAULT_LAZY_IMPUTER = LazyDeltaImputer
