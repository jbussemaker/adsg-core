# Architecting Patterns

System architecture patterns are defined in
*Selva, Cameron and Crawley, "Patterns in System Architecture Decisions", 2016, DOI: 10.1002/sys.21370*,
and cover a range of common types of choices that occur when designing system architectures.

This software package uses a more general way to model choices based on connections from source to target nodes,
represented using matrices. Nodes can have constraint in terms of nr of connections and whether repetition is allowed or
not. Model configuration notation:
- Source/target nodes: `n` @ `m` [(rep)] (`n` nodes where each node accepts `m` connections), where `m` can be:
  - A fixed number: `m` (e.g. `0`, `1`, `5`)
  - A set of fixed numbers: `m,o,p` (e.g. `0,1`, `0,1,3`)
  - A minimum amount: `m+` (e.g. `0+`, `1+`)
  - A range of numbers (inclusive): `m..o` (e.g. `0..2`, `1..5`)
  - (rep) specifies whether repeated connections to/from that node are allowed, by default they are not allowed
- Excluded edges: `(i_src,i_tgt),...` (e.g. `(0,1),(4,3)`)
- Existence schemas: any override of a combination of source/target node definitions

The architecting patterns are represented by connection choices as follows:

| Pattern                            | Parameters      | Source nodes             | Target nodes        | Excluded edges    |
|------------------------------------|-----------------|--------------------------|---------------------|-------------------|
| Combining                          | `n_tgt`         | `1 @ 1`                  | `n_tgt @ 0,1`       |                   |
| Unordered combining                | `n_take, n_tgt` | `1 @ n_take (rep)`       | `n_tgt @ 0+ (rep)`  |                   |
| Unordered non-replacing combining  | `n_take, n_tgt` | `1 @ n_take`             | `n_tgt @ 0,1`       |                   |
| Assigning                          | `n_src, n_tgt`  | `n_src @ 0+`             | `n_tgt @ 0+`        |                   |
| Assigning (injective)              | `n_src, n_tgt`  | `n_src @ 0+`             | `n_tgt @ 0,1`       |                   |
| Assigning (surjective)             | `n_src, n_tgt`  | `n_src @ 0+ or 1+`       | `n_tgt @ 1+`        |                   |
| Assigning (bijective)              | `n_src, n_tgt`  | `n_src @ 0+ or 1+`       | `n_tgt @ 1`         |                   |
| Assigning (repeatable)             | `n_src, n_tgt`  | `n_src @ 0+ (rep)`       | `n_tgt @ 0+  (rep)` |                   |
| Assigning (repeatable, surjective) | `n_src, n_tgt`  | `n_src @ 0+ or 1+ (rep)` | `n_tgt @ 1+  (rep)` |                   |
| Partitioning                       | `n_src, n_tgt`  | `n_src @ 0+`             | `n_tgt @ 1`         |                   |
| Partitioning (covering)            | `n_src, n_tgt`  | `n_src @ 0+`             | `n_tgt @ 1+`        |                   |
| Downselecting                      | `n_tgt`         | `1 @ 0+`                 | `n_tgt @ 0,1`       |                   |
| Connecting                         | `n`             | `n @ 0+`                 | `n @ 0+`            | `(i,j) if i >= j` |
| Connecting (directed)              | `n`             | `n @ 0+`                 | `n @ 0+`            | `(i,j) if i == j` |
| Permuting                          | `n`             | `n @ 1`                  | `n @ 1`             |                   |

Compared to the patterns defined by Selva et al., the following extensions are made:
- For the assigning pattern:
  - The "function" aspect (i.e. each source node has 1 connection) of injective, surjective
    and bijective problems are dropped, in order to increase the problem size
    and not be constrained by `n_src` relative to `n_tgt`
  - Also versions where each source node needs at least one connection are considered as an assigning pattern
- Repeatable versions of the assigning versions are added in order to increase the number of test problems
  - Note that any repeatable node that has an upper bound of 0 or 1 in effect is non-repeatable
- Two additional patterns are added:
  - Unordered combining: take combinations of length `n_take` from a set, ignoring permutations e.g. 2 from ABC: AA, AB, AC, BB, BC, CC
    - This pattern is the same as a combination pattern if `n_take = 1`
    - It can also be seen as several combining pattern instances merged into one where the order in decision selection
      is not relevant (hence: unordered)
  - Unordered non-replacing combining: same as the above however values can't be taken again: e.g. 2 from ABC: AB, AC, BC
    - This pattern can be seen as a downselecting problem with a fixed size of the selected set
  - These two patterns can be seen as versions of the combining, permuting and assigning patterns,
    where the order of connections is not relevant (i.e. AB has the same fitness as BA)
- Each pattern can also be defined as its transpose (i.e. sources and targets swapped)

As can be seen in the table above, there are normally only a few values used for the number of node connections:

| src↓, tgt→ | 1         | 0,1       | n             | 0+                                           | 1+                                                | 0+ (rep)               | 1+ (rep)                           |
|------------|-----------|-----------|---------------|----------------------------------------------|---------------------------------------------------|------------------------|------------------------------------|
| 1          | Permuting | Combining | (1)           | Assigning (bijective) == partitioning        | (2)                                               | (3)                    | (3)                                |
| 0,1        |           | (4)       | UNR Combining | Assigning (injective), downselecting         | (2)                                               | (3)                    | (3)                                |
| n          |           |           |               | Unordered combining                          |                                                   | (3)                    | (3)                                |
| 0+         |           |           |               | Assigning, connecting, connecting (directed) | Assigning (surjective) == partitioning (covering) | (3)                    | (3)                                |
| 1+         |           |           |               |                                              | Assigning (surjective)                            | (3)                    | (3)                                |
| 0+ (rep)   |           |           |               |                                              |                                                   | Assigning (repeatable) | Assigning (repeatable, surjective) |
| 1+ (rep)   |           |           |               |                                              |                                                   |                        | Assigning (repeatable, surjective) |

(1) only possible if `n_src = n*n_tgt`, in which case it is effectively `n` independent permuting patterns.
(2) effectively permuting if `n_tgt = n_src`, otherwise only relevant if `n_src > n_tgt`.
(3) effectively their non-repeatable versions.
(4) combining with the possibility of not choosing an option.

## Pattern-Specific Encoders

Pattern-specific encoders are selected by a matching procedure: for a given model configuration, each pattern encoder
reports whether it is compatible with that configuration. A pattern encoder is compatible if for a configuration or
its transpose, the pattern encoder reports that it is compatible with each of the effective configurations. An effective
configuration is a model configuration where for each existence schema the configuration has been modified as to what
that existence schema effectively means.

The following logic is implemented (not that the transpose is also matched):

| Pattern encoder     | Variant                | Source nodes       | Target nodes        | Additional constraints       | Matched patterns                                           |
|---------------------|------------------------|--------------------|---------------------|------------------------------|------------------------------------------------------------|
| Combining           |                        | `1 @ 1`            | `n_tgt @ 0,1`       |                              | Combining                                                  |
| Combining           | Collapsed              | `1 @ * (rep)`      | `1 @ * (rep)`       |                              | Combining                                                  |
| Unordered combining | With replacement       | `1 @ n_take (rep)` | `n_tgt @ 0+ (rep)`  |                              | Unordered combining                                        |
| Unordered combining |                        | `1 @ n_take`       | `n_tgt @ 0,1`       | `n_take <= n_tgt`            | Combining, unordered non-replacing combining               |
| Assigning           |                        | `n_src @ 0+`       | `n_tgt @ 0+`        |                              | Assigning                                                  |
| Assigning           | Surjective             | `n_src @ k+`       | `n_tgt @ 1+`        | `k >= 0`                     | Surjective assigning, covering partitioning                |
| Assigning           | Repeatable             | `n_src @ 0+ (rep)` | `n_tgt @ 0+  (rep)` |                              | Repeatable assigning                                       |
| Assigning           | Repeatable, surjective | `n_src @ k+ (rep)` | `n_tgt @ 1+  (rep)` | `k >= 0`                     | Surjective repeatable assigning                            |
| Partitioning        |                        | `n_src @ k+`       | `n_tgt @ 0,1 or 1`  | `k >= 0 && k*n_src <= n_tgt` | Partitioning, downselecting, injective/bijective assigning |
| Connecting          |                        | `n @ 0+`           | `n @ 0+`            | Excluded: `(i,j) if i >= j`  | Connecting                                                 |
| Connecting          | Directed               | `n @ 0+`           | `n @ 0+`            | Excluded: `(i,j) if i == j`  | Directed connecting                                        |
| Permuting           |                        | `n @ 1`            | `n @ 1`             |                              | Permuting                                                  |
