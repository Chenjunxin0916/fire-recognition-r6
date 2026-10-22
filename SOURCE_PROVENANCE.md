# Frozen source provenance

The initial revised-split three-arm execution used release `split_v2_classifier_preflight_B_20261021_v2`. Its accepted evaluation report records source SHA-256 `7623b98ab495734157266d1221085f643b311d6aebb435b4219e0e659b2b709c` and a 1,100-image frozen input snapshot. The later HN/random control used release `hn_random_control_split_v2_20261022_go_v2`; its plan records source SHA-256 `16a736bec72b8a12106c4a78aaec477880ab5b51a4d446b4fb1dc1d5ba839c34`.

The later release's `provenance/execution_snapshot` supplied this candidate's main `src/`, selected `scripts/`, accepted configurations, and requirement records. The two frozen snapshots differ in six implementation files; exact copies of the earlier versions are under `formal_release_source/` with their original relative paths. This prevents the later implementation from being misrepresented as the byte-identical code of the initial run.

The candidate is curated. It excludes run outputs and unrelated scripts/configurations, retains public package namespace stubs, and adds previously reviewed copyright/attribution comments to HN and Focal source. Those comments do not alter executable Python syntax trees. The included analysis scripts read private, separately retained input files; they are source provenance rather than a self-contained numerical replication bundle.

`SOURCE_MANIFEST.csv` lists every candidate file other than itself, with size and SHA-256. The local build log and mapping to the two frozen source trees are retained in the author's internal project. Neither frozen source SHA above is a hash of this curated package.
