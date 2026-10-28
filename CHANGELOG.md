# Changelog

## [3.0.0](https://github.com/GetActionSpace/mcp-context-server/compare/v2.2.2...v3.0.0) (2026-10-28)


### ⚠ BREAKING CHANGES

* ENABLE_EMBEDDING_COMPRESSION defaults to true. New deployments will store embeddings in the bit-packed compressed format and write a singleton compression_metadata row on first start; the configured (provider, bits, variant, seed, dim) tuple becomes load-bearing and immutable thereafter. Existing deployments are unaffected at runtime because their compression_metadata table does not exist until a compression-aware start, but operators upgrading to v3.0.0 who want to retain fp32 storage MUST set ENABLE_EMBEDDING_COMPRESSION=false before first start; otherwise the server will initialize compressed storage using the default seed (0). In multi-pod Kubernetes deployments all pods MUST inherit the SAME COMPRESSION_SEED via a shared ConfigMap; changing the seed after compressed data exists corrupts every decode/search with no recovery path besides restoring from backup.
* The public type of context_entries.id and all foreign keys referencing it has changed from a 64-bit integer to a UUIDv7 value. MCP tools now accept and return 32-character lowercase hex strings, and the 36-character hyphenated form is also accepted at the boundary. Existing databases created with the integer-PK layout are NOT auto-migrated by the server, and operators MUST run the new mcp-context-server-migrate CLI against a backup of the source database before pointing the upgraded server at it. See docs/MIGRATION-v2-to-v3.md for the full step-by-step procedure. Clients that parse or persist context IDs as integers MUST be updated to handle string UUIDv7 values.
* the project license changed from the MIT License to the Elastic License 2.0; earlier published releases remain under the MIT License, and providing the software to third parties as a hosted or managed service that exposes a substantial set of its functionality now requires a commercial agreement with the licensor.

### Features

* add a concurrency option to mcp-context-ingest ([#137](https://github.com/GetActionSpace/mcp-context-server/issues/137)) ([47f429d](https://github.com/GetActionSpace/mcp-context-server/commit/47f429d75494dec1c5019d8c6940f26602e441b7))
* add access-control schema with server-stamped ownership and visibility ([1c0a3d9](https://github.com/GetActionSpace/mcp-context-server/commit/1c0a3d9e228261459cfc2ef93fff1d8dcef90d0c))
* add access-control schema with server-stamped ownership and visibility ([30f4ef3](https://github.com/GetActionSpace/mcp-context-server/commit/30f4ef323900cf89c3e87119dc67542461457857))
* add configurable summary inclusion for get_context_by_ids ([800eb21](https://github.com/GetActionSpace/mcp-context-server/commit/800eb21098d2aa21a329af98ac2af7f3c23bd971))
* add corpus sync tools and the mcp-context-ingest command ([#134](https://github.com/GetActionSpace/mcp-context-server/issues/134)) ([fbd154e](https://github.com/GetActionSpace/mcp-context-server/commit/fbd154e06f677238dbb60a9e2f879fa2642adcbf))
* add entry sharing tools, visibility read-back and a read-only anonymous principal ([#138](https://github.com/GetActionSpace/mcp-context-server/issues/138)) ([e52173c](https://github.com/GetActionSpace/mcp-context-server/commit/e52173c40b4511b7502f6eae53d759005abbaf69))
* add grep/navigate/read context tools with index_tree node summaries ([2304bc6](https://github.com/GetActionSpace/mcp-context-server/commit/2304bc65ee48299d3d1935a773479b111d53aa14))
* add JWT authentication provider with IdP-agnostic claims mapping ([9332776](https://github.com/GetActionSpace/mcp-context-server/commit/9332776145aa988436bacb7f9ac402f31bf5ca92))
* add JWT authentication provider with IdP-agnostic claims mapping ([01c29c0](https://github.com/GetActionSpace/mcp-context-server/commit/01c29c0c7431eaa4d3a1c3c6dc3f67d237aeee7b))
* add migration CLI --re-embed flag for switching embedding models ([238aa8e](https://github.com/GetActionSpace/mcp-context-server/commit/238aa8ebf26a2b9c42df75ab4fae097eae9ffb88))
* add optional pagination to list_threads ([51a1e30](https://github.com/GetActionSpace/mcp-context-server/commit/51a1e300560f6143a10e4828373d323520a5986f))
* add query-side instruction support for instruct-aware embedding models ([8408cd3](https://github.com/GetActionSpace/mcp-context-server/commit/8408cd3ff2bcfb497a88d03d764a1aeb31534fff))
* add TurboQuant embedding compression ([989cde1](https://github.com/GetActionSpace/mcp-context-server/commit/989cde1c2f07e157a95f4edcba2d8656b68a4a00))
* allow overriding the provider existingSecret data key name in the chart ([c23308e](https://github.com/GetActionSpace/mcp-context-server/commit/c23308e2943f5e61bd786f65a70105fc8f78b078))
* auto-enable search tools by default ([8045194](https://github.com/GetActionSpace/mcp-context-server/commit/8045194b483823c70d48fe6710386c1bf14f23f4))
* auto-set search_path on every pooled asyncpg connection ([26639f3](https://github.com/GetActionSpace/mcp-context-server/commit/26639f33d83bc2367319d0e6946661b66b7d5092))
* enforce access scoping with PostgreSQL row-level security ([#132](https://github.com/GetActionSpace/mcp-context-server/issues/132)) ([99d103a](https://github.com/GetActionSpace/mcp-context-server/commit/99d103ab833799d6ccedad235b057917e467f8d4))
* expose compression configuration via startup log and get_statistics ([72033be](https://github.com/GetActionSpace/mcp-context-server/commit/72033be1b4b442b4d955d4aa4b429a578004de5a))
* migrate context-entry primary keys to UUIDv7 ([31877ae](https://github.com/GetActionSpace/mcp-context-server/commit/31877aef6f4b949b1c622b062c82e2fafbc76a34))
* overlap store/update generation and guard concurrent updates ([612e77c](https://github.com/GetActionSpace/mcp-context-server/commit/612e77c16707bbc6fc13044a7701d38b51a0f11c))
* relicense under the Elastic License 2.0 ([6ae2e42](https://github.com/GetActionSpace/mcp-context-server/commit/6ae2e42a3797a5a6b6f39d17028593123935ae03))
* report embeddings_size_mb and fix get_statistics on PostgreSQL ([4406a11](https://github.com/GetActionSpace/mcp-context-server/commit/4406a11b3433776bfcd56ac255fc3b48517875c0))
* scope reads and writes to the calling principal ([#131](https://github.com/GetActionSpace/mcp-context-server/issues/131)) ([d4f1f89](https://github.com/GetActionSpace/mcp-context-server/commit/d4f1f891b6e12f29e69ac754b49bdfe0bef976a5))
* upgrade to FastMCP 4.0.10 and MCP Python SDK 2.2 ([#128](https://github.com/GetActionSpace/mcp-context-server/issues/128)) ([cd39362](https://github.com/GetActionSpace/mcp-context-server/commit/cd39362997f2696686de5b2049a7a6f603ea4163))
* warn on oversized pool behind a Supabase Session Pooler ([f6b4691](https://github.com/GetActionSpace/mcp-context-server/commit/f6b4691663f6bcabf43ab4fb39fb18294d79fe35))


### Bug Fixes

* acquire the schema-init advisory lock under the migration timeout ([517ad7e](https://github.com/GetActionSpace/mcp-context-server/commit/517ad7edefc30ef24d1e812fa4d0f1786ff343b4))
* align embedding provisioning gates and guard probes with bare-name resolution ([df76b5e](https://github.com/GetActionSpace/mcp-context-server/commit/df76b5e8f6786d0a1b938b16631df80b036e0227))
* align SQLite/PostgreSQL behavior and harden edge cases ([c284a9b](https://github.com/GetActionSpace/mcp-context-server/commit/c284a9b231c95b880504bfe61dff66826b8960f2))
* align the hybrid all-modes-failed response with the documented shape ([ce94a9d](https://github.com/GetActionSpace/mcp-context-server/commit/ce94a9d38bd27c9868a924b6c1c96174929dc801))
* always clear stale index_tree node summaries on a text-change update ([488dfa9](https://github.com/GetActionSpace/mcp-context-server/commit/488dfa9aa199256de47c0c0a2a0072405a820d68))
* always return at least one chunk from split_text ([8f91abb](https://github.com/GetActionSpace/mcp-context-server/commit/8f91abb4c0667b810c2ec72a96a70d9a62b73a3b))
* attach explain-gated stats to remaining search validation-error responses ([2cda68e](https://github.com/GetActionSpace/mcp-context-server/commit/2cda68edfe518fdb1b80197f293bec0612150c3c))
* await every compression encode before propagating a failure ([dd3c104](https://github.com/GetActionSpace/mcp-context-server/commit/dd3c104c63e14f307b2a315309e1bd5f531fc5f6))
* bound metadata filter and grep pattern inputs at the tool boundary ([36fcb12](https://github.com/GetActionSpace/mcp-context-server/commit/36fcb12f75369e57f1d7eaf8fc3abbf530e72b25))
* bound POSTGRESQL_PORT to a valid TCP port range ([370167a](https://github.com/GetActionSpace/mcp-context-server/commit/370167a50d1ffc3d7ae9fd76b6a674ffe4107dfe))
* bound search pagination offset and clamp overfetch windows ([604c9cd](https://github.com/GetActionSpace/mcp-context-server/commit/604c9cdcf0c3007d5642f639080ce94324f4990f))
* bound the PostgreSQL pool size fields at the configuration layer ([31841f2](https://github.com/GetActionSpace/mcp-context-server/commit/31841f2d62a949015103781a30b7c455af584e62))
* bound tool-boundary inputs and correct response contracts ([7058183](https://github.com/GetActionSpace/mcp-context-server/commit/7058183b5aa55680b082966401ef69daa435f6d9))
* bracket IPv6 host literals in the built PostgreSQL DSN ([72e13cf](https://github.com/GetActionSpace/mcp-context-server/commit/72e13cfd8aedd8e86b8036a0bfae7f0eed97e009))
* canonicalize date filter parameters for cross-backend parity ([51db9ec](https://github.com/GetActionSpace/mcp-context-server/commit/51db9ec3b81022175e0abae4fb5c7bec104af2d3))
* cap node-id slug segments below the PostgreSQL index-tuple ceiling ([2602eba](https://github.com/GetActionSpace/mcp-context-server/commit/2602eba5c71e84c378805f552d157e49cd743523))
* cap tags filters and metadata membership lists at 100 members ([2576023](https://github.com/GetActionSpace/mcp-context-server/commit/257602383cfe2845878c85697d1512e398b979cc))
* cap the delete_context ID list at the tool boundary ([d63396b](https://github.com/GetActionSpace/mcp-context-server/commit/d63396b3160a6fee0ba15d5b1005d47587a2db77))
* charge SQLite connection-creation faults and stop charging locked reads ([3759087](https://github.com/GetActionSpace/mcp-context-server/commit/37590875561c9e6d6e7f6acf5a5cc35f8dd50002))
* chunk ID-list binds and enforce the batch ID caps ([cf2a221](https://github.com/GetActionSpace/mcp-context-server/commit/cf2a22198d811d45c7b01c9842fc9385b928fbcb))
* clamp the numeric filter float8 underflow band to match SQLite ([3fbf0e4](https://github.com/GetActionSpace/mcp-context-server/commit/3fbf0e4dacb893d8b3a54dfdaec9ec1f4a74d5ef))
* classify a malformed DSN as a configuration error in the pgvector pre-check ([79faae0](https://github.com/GetActionSpace/mcp-context-server/commit/79faae0ccb9061d72cc097e5891d9c02c4d619ee))
* classify FTS engine failures by catalog provenance and keep boot diagnostics fault-tolerant ([38db489](https://github.com/GetActionSpace/mcp-context-server/commit/38db4898a887d3305222ca7b77f43d1d4a899925))
* classify PostgreSQL acquire timeouts by elapsed wait ([7943531](https://github.com/GetActionSpace/mcp-context-server/commit/7943531851a460bea02aca9e9bf151dedc2c3ef6))
* classify provider errors precisely and align summary provider docs ([adbe5d6](https://github.com/GetActionSpace/mcp-context-server/commit/adbe5d643b07766f1803c3fba3fe5f0b60ad30eb))
* clear stale summary on text-change batch update without a provider ([810d60a](https://github.com/GetActionSpace/mcp-context-server/commit/810d60a1c86af390cd8a903bb8d86d8fa9aa18ce))
* compare bearer tokens as UTF-8 bytes for non-ASCII tokens ([783f254](https://github.com/GetActionSpace/mcp-context-server/commit/783f2545a19270c7a2ce33285eb020e145367381))
* compensate the preserved-summary count on discarded batch entries ([edbb504](https://github.com/GetActionSpace/mcp-context-server/commit/edbb50461c46506f14e88fb2d4c9555ad3bff5a7))
* complete search diagnostics for timing, backend, and degraded modes ([c1120e2](https://github.com/GetActionSpace/mcp-context-server/commit/c1120e2ef1c9bb8c69553ca89aa6dfab782194a4))
* complete search stats parity on degraded and validation-error responses ([7452586](https://github.com/GetActionSpace/mcp-context-server/commit/745258652a5d1066222f239c4dee2a4429ff5c67))
* constrain the SQLite thread delete to the cleanup snapshot ([d752e63](https://github.com/GetActionSpace/mcp-context-server/commit/d752e635c8fe32297a80484159fc9e8bc71549d5))
* correct decompress rebuild, metadata operators, and cleanup scoping ([4ab35be](https://github.com/GetActionSpace/mcp-context-server/commit/4ab35be22639101eb2881fe32c323101b9cfaa2c))
* correct nested metadata key handling and bound float filter SQL ([0c194cb](https://github.com/GetActionSpace/mcp-context-server/commit/0c194cbd44db7e9da5a195a0b7d4a3bbe90d3508))
* correct PostgreSQL schema-qualified DDL, vector provisioning, and guard ordering ([734c995](https://github.com/GetActionSpace/mcp-context-server/commit/734c995d738319c4a393a306b1575ab175953aed))
* correct the semantic and hybrid tool-registration descriptions ([6643649](https://github.com/GetActionSpace/mcp-context-server/commit/6643649bdab58af6787285fb404dc297ecc9abcb))
* count a literal grep match toward the cap only after the terminator check ([2f10fc2](https://github.com/GetActionSpace/mcp-context-server/commit/2f10fc26d99056c311fd1e9e974411b2e423f5d1))
* degrade a boolean FTS query with a non-integer NEAR distance to a term match ([f68f69d](https://github.com/GetActionSpace/mcp-context-server/commit/f68f69de3cf771a8d1716d220aa7ebb44cab9fd6))
* degrade a leading-asterisk boolean FTS query to a safe term match ([307af02](https://github.com/GetActionSpace/mcp-context-server/commit/307af0285936168e065b2ad15e21b91c5cc70da5))
* degrade malformed SQLite FTS boolean query to a safe term match ([2073ca5](https://github.com/GetActionSpace/mcp-context-server/commit/2073ca59a5a5fb910030a316a341b4037a462487))
* delete exactly the snapshotted ids in SQLite batch deletes ([2c4c550](https://github.com/GetActionSpace/mcp-context-server/commit/2c4c55056545359bfb6929d7f4c65c42e9ed6047))
* derive the Helm container port from the server bind port ([259b555](https://github.com/GetActionSpace/mcp-context-server/commit/259b5551a6cc2cf9191534e97f1c7135b584dec9))
* detect cross-host codebook divergence and guard compress dimension ([1206d21](https://github.com/GetActionSpace/mcp-context-server/commit/1206d21ca3c637f923cef50deef2552236ef367d))
* disable the LLM SDK internal retry in the summary providers ([9e892a4](https://github.com/GetActionSpace/mcp-context-server/commit/9e892a49246e46a32d00d8e88526bbede6be090a))
* drain the executor future on every SQLite transaction-boundary hop ([bac6858](https://github.com/GetActionSpace/mcp-context-server/commit/bac68587fe3c924001eea4b96ebae65fdab4be98))
* drain the executor future on the SQLite read-execution hop ([a3bb3dc](https://github.com/GetActionSpace/mcp-context-server/commit/a3bb3dc6ae1b0592c89a1e29951644e6a302d77a))
* embed documents in batches with a per-batch budget and fair scheduling ([#140](https://github.com/GetActionSpace/mcp-context-server/issues/140)) ([c45f0be](https://github.com/GetActionSpace/mcp-context-server/commit/c45f0be75519c5c68e98c8fd318115421395bcc7))
* enforce SQLite/PostgreSQL parity in metadata filters and FTS queries ([ddc2f28](https://github.com/GetActionSpace/mcp-context-server/commit/ddc2f28f55fe081522ec172605a6ed64bdb20241))
* exit with the configuration code when settings validation fails at import ([4f30b1a](https://github.com/GetActionSpace/mcp-context-server/commit/4f30b1a80606ced74e5dbd9fcaa41ae380bea348))
* exit with the configuration code when the auth token is missing ([725ec9f](https://github.com/GetActionSpace/mcp-context-server/commit/725ec9f49848a5b6702fc289d107b0e84a12b148))
* filter the non-atomic update-batch existence errors by original index ([07d4520](https://github.com/GetActionSpace/mcp-context-server/commit/07d4520464a037e1c0ed81cca546b50b78a27f7d))
* floor migration statement timeouts at one millisecond ([a6164b3](https://github.com/GetActionSpace/mcp-context-server/commit/a6164b31f6e1dbf6e9258c919f0928f8394726cc))
* floor the derived statement_timeout at one millisecond ([0c063ea](https://github.com/GetActionSpace/mcp-context-server/commit/0c063ea9f6fef8dd9f7f4297e7ec227bd10a1c27))
* fold an empty summary effort env value to None ([916efdc](https://github.com/GetActionSpace/mcp-context-server/commit/916efdc25c2a49aa2041baec498372a777d7af11))
* force stateful sessions for the SSE transport ([891a12a](https://github.com/GetActionSpace/mcp-context-server/commit/891a12acc3bc41a977de1f9b864f0a4665f2be84))
* forward the explicit tool name to the FastMCP registration ([26090cd](https://github.com/GetActionSpace/mcp-context-server/commit/26090cd25f6c32cfe08baa129194157d929fefd6))
* freeze the source table before streaming in the compression CLI ([cdca1c8](https://github.com/GetActionSpace/mcp-context-server/commit/cdca1c82cf09a91c293faa4466a0d30098d142b1))
* gate compression provisioning on embedding generation ([8b0ab75](https://github.com/GetActionSpace/mcp-context-server/commit/8b0ab75b5b659f6f060c0f2b4c5b2abe6fa0eafe))
* gate delete embedding cleanup on table presence, not runtime toggles ([3678307](https://github.com/GetActionSpace/mcp-context-server/commit/3678307b7e118649a9bd2ab6171c8c812c605e2f))
* gate delete_context embedding cleanup on generation or compression ([1b642fb](https://github.com/GetActionSpace/mcp-context-server/commit/1b642fbd9cd3e4e9241d2e94dffe3f980cd10ee0))
* gate pgvector provisioning on generation and the active payload format ([8bc9be6](https://github.com/GetActionSpace/mcp-context-server/commit/8bc9be62455330ef5c0521c940e9613ad6e55027))
* gate the SQLite full-text-search operator-bareword drop on the configured language ([1aea4d9](https://github.com/GetActionSpace/mcp-context-server/commit/1aea4d9acf4536d9c5702b9da611e596f4ca8132))
* give hyphenated hybrid terms phrase semantics on PostgreSQL ([34c6300](https://github.com/GetActionSpace/mcp-context-server/commit/34c630018c69921e05765e0ea981d04717a15cd3))
* guard populated fp32 stores before the generation-off compression skip ([43c6011](https://github.com/GetActionSpace/mcp-context-server/commit/43c60119bf18a2182a0b53cece921e6b0797f5b7))
* guard the full sealed compression tuple before backfilling embeddings ([21e5a7e](https://github.com/GetActionSpace/mcp-context-server/commit/21e5a7e22cdab489a2416b45b3f6b7bb06aa9a87))
* guard the open-breaker resolution against a completed future ([7901a4b](https://github.com/GetActionSpace/mcp-context-server/commit/7901a4bfc046b9b355d240156a9c4fe977607fa2))
* harden backend connection lifecycle and fault accounting ([84a2a31](https://github.com/GetActionSpace/mcp-context-server/commit/84a2a3140b64277a4d9bf599e092eadbf24952de))
* harden backend connection lifecycle and fault classification ([ba9da54](https://github.com/GetActionSpace/mcp-context-server/commit/ba9da547c99012b38f126d42ddf3b2c143e862e4))
* harden grep/navigate generation against regex blowups and outline edge cases ([1649790](https://github.com/GetActionSpace/mcp-context-server/commit/164979008eff8c81b83befe4d8bd48537ef5126c))
* harden metadata index configuration validation and strict-mode failure ([1d875b7](https://github.com/GetActionSpace/mcp-context-server/commit/1d875b78d26f2f8aaaeb8eaf8ae65a2bdb443281))
* harden PostgreSQL and timing configuration at the boundary ([125293b](https://github.com/GetActionSpace/mcp-context-server/commit/125293b5a289952d9fb531aa1a4ecffefd981e9e))
* harden storage integrity across dedup, updates, batch, and backends ([6c9619d](https://github.com/GetActionSpace/mcp-context-server/commit/6c9619d774e8702ff65feb333ef71bc47da6da00))
* harden the migrate CLI against invalid settings and FTS-less targets ([560f89f](https://github.com/GetActionSpace/mcp-context-server/commit/560f89f984f9965adc0cb2c60491e4df4e105256))
* include content_hash in the cross-backend unstorable-string pre-check ([a0334f2](https://github.com/GetActionSpace/mcp-context-server/commit/a0334f24d3091c8a07dc43f086ec2cc66717f974))
* initialize a pgvector-less migration target without the vector extension ([0a51c1a](https://github.com/GetActionSpace/mcp-context-server/commit/0a51c1a666b7e4b0fa5d6bc0fea0a1c605cd1b70))
* isolate batch reconcile embedding failure to the failing entry ([59145fe](https://github.com/GetActionSpace/mcp-context-server/commit/59145feb4fcdf12a8db11d949727f69ee927c9ea))
* join the executor closure before a cancelled SQLite transaction unwinds ([15a0de5](https://github.com/GetActionSpace/mcp-context-server/commit/15a0de5d221fe945fa63529846944800c9df5308))
* keep control-flow rejections out of the failed-query read metric ([baab1a2](https://github.com/GetActionSpace/mcp-context-server/commit/baab1a2d0f6ab3b839224a4131d5bb5598a17ea2))
* leave an already-bracketed IPv6 host literal unmodified in the DSN ([e0b1d6c](https://github.com/GetActionSpace/mcp-context-server/commit/e0b1d6c26ee92f23a7fef924878ca85e09678cc2))
* list the metadata equality filter in the served full-text-search description ([6feb450](https://github.com/GetActionSpace/mcp-context-server/commit/6feb450214db2a5796a88a493cba02d0674eb059))
* list the real embedding tables in the PostgreSQL dimension-mismatch message ([758db4b](https://github.com/GetActionSpace/mcp-context-server/commit/758db4bccd5138ffa9d1d2a7601154581b8491b0))
* load the sqlite-vec extension whenever the package is installed ([7b3da6d](https://github.com/GetActionSpace/mcp-context-server/commit/7b3da6ddfe02466dfa439ec58e9e72ae06250901))
* lock the parent row during transactional presence checks on PostgreSQL ([7b0049a](https://github.com/GetActionSpace/mcp-context-server/commit/7b0049aa831634c62fd255e8d6b5f83c2ba17038))
* make cross-backend migration CLI schema-aware and source-tolerant ([0cb168b](https://github.com/GetActionSpace/mcp-context-server/commit/0cb168b90167f805709a5ace028eff4e69b07abb))
* make LOG_LEVEL govern FastMCP logging and pin lax input validation ([464e606](https://github.com/GetActionSpace/mcp-context-server/commit/464e606721603f9106a8c0a5e308747f4cf6c4bb))
* make LOG_LEVEL govern the uvicorn logger tree and silence FastMCP when disabled ([4203f34](https://github.com/GetActionSpace/mcp-context-server/commit/4203f349ceb8a5cd4ad52a6ff19b465a0f9f4bc6))
* make ranked search deterministic and correct FTS handling ([87713a8](https://github.com/GetActionSpace/mcp-context-server/commit/87713a878573144211627fc19829084d28a19c06))
* make tags a set per entry and align repository ordering across backends ([adf726d](https://github.com/GetActionSpace/mcp-context-server/commit/adf726d70b939fae0067d12e821336113fe9f80f))
* make v2-to-v3 migration robust across backends and settings ([94600a7](https://github.com/GetActionSpace/mcp-context-server/commit/94600a759995ac16eeb48737607e8d54941207ad))
* map the numeric filter overflow guard to the true float8 boundary ([1a7aa65](https://github.com/GetActionSpace/mcp-context-server/commit/1a7aa65e33b8de1d091722cd44ed70cac3c86325))
* match SQLite for high-magnitude float metadata filters on PostgreSQL ([4665a39](https://github.com/GetActionSpace/mcp-context-server/commit/4665a394297bbee2856266dbd6e2811103f5b559))
* match SQLite for numeric metadata filters and reject non-finite values ([0ac460e](https://github.com/GetActionSpace/mcp-context-server/commit/0ac460ed0ccd1443c585dde7a346af52cd51672c))
* normalize base64 image payloads strictly and enforce the image count limit ([c773417](https://github.com/GetActionSpace/mcp-context-server/commit/c7734175817427fadf5c34e790f01c9aa2a9f7b9))
* normalize context IDs before ctx.info log emission ([5db4b79](https://github.com/GetActionSpace/mcp-context-server/commit/5db4b79fe30d0d93c9951b4c6d8a85ce29f3de83))
* offload large-entry outline and line parsing off the event loop ([ee48c34](https://github.com/GetActionSpace/mcp-context-server/commit/ee48c34593b9befa2c9387fedbd0502c539683f2))
* omit the orphaned chart OpenAI Secret key when an existingSecret is used ([0cdeaf1](https://github.com/GetActionSpace/mcp-context-server/commit/0cdeaf1507116ee71738eb2edddee0bf55cf3e4c))
* open sqlite databases from POSIX absolute URLs and URI-special paths ([df80022](https://github.com/GetActionSpace/mcp-context-server/commit/df8002256badedbafa91252f4f211cfe240c0d7a))
* prevent FTS metadata-key corruption and offload large-entry CPU work ([bd46319](https://github.com/GetActionSpace/mcp-context-server/commit/bd46319e25bbb9876b9d95261975a63d2bd97f5a))
* prevent run_generation task leak and depth-shifted outline node ids ([7636401](https://github.com/GetActionSpace/mcp-context-server/commit/7636401f78655d0c0ad3562f901700e2a377d688))
* probe a compression-safe marker for the chunking migration on PostgreSQL ([0e7762c](https://github.com/GetActionSpace/mcp-context-server/commit/0e7762cf87129e1cc3ae8552cee57820a1a1b0e8))
* probe for more matches before flagging a budget-stopped grep scan truncated ([ffdc708](https://github.com/GetActionSpace/mcp-context-server/commit/ffdc708a16c4a41250be6586ff59da229e48e187))
* probe the version token when retaking the navigate snapshot ([5b8653b](https://github.com/GetActionSpace/mcp-context-server/commit/5b8653be77cbfae324b878bda54e7137dc8bd799))
* propagate non-absence read errors from the index-node repository ([dc318fd](https://github.com/GetActionSpace/mcp-context-server/commit/dc318fdc4f26af35317d12b6e36ebf0c32141d2c))
* protect migration CLI against data loss and unstorable input ([4d1a845](https://github.com/GetActionSpace/mcp-context-server/commit/4d1a845c8c792f1bb245ff7bbb727e0d745839a5))
* protect the migration CLI against timeouts, oversized values, and schema drift ([639f324](https://github.com/GetActionSpace/mcp-context-server/commit/639f32494c98195341cd53d32242e2a62d7b220e))
* provision a migrated PostgreSQL target's full-text search from source presence ([7e9842c](https://github.com/GetActionSpace/mcp-context-server/commit/7e9842cbe75f02f9e94b4081881d6177b42c9276))
* provision the database file only for the SQLite backend ([9fb49e8](https://github.com/GetActionSpace/mcp-context-server/commit/9fb49e8e3db1a9e14d86e980761023e283698301))
* qualify metadata column by alias and provision dedup index in base schema ([1e8aea8](https://github.com/GetActionSpace/mcp-context-server/commit/1e8aea8f7404bcec109c7f0cfbdc32db6bf8b76b))
* quote metadata index identifiers and drop base-schema metadata indexes ([9b4b0de](https://github.com/GetActionSpace/mcp-context-server/commit/9b4b0de53f202cdf5775bc8a187d05c6af4e1657))
* quote the configured schema in patch_metadata's runtime function call ([02c3e67](https://github.com/GetActionSpace/mcp-context-server/commit/02c3e67be57c0bef4b0d5d7ca213ca49b5571dd7))
* quote the schema in the orphan metadata-index drop ([d33ad8d](https://github.com/GetActionSpace/mcp-context-server/commit/d33ad8dc9b683a974e3fa557bb51cf7ffe2c3cfc))
* raise ConfigurationError on embedding provider 4xx in is_available ([88210e4](https://github.com/GetActionSpace/mcp-context-server/commit/88210e497695a681ea1d183f23018dc18e4622d6))
* raise dependency floors to releases that fix known vulnerabilities ([05bc942](https://github.com/GetActionSpace/mcp-context-server/commit/05bc94216392a4bd53a5282b5d510f4f651406bf))
* re-assert the dedup decision in the store UPDATE predicate ([38711c8](https://github.com/GetActionSpace/mcp-context-server/commit/38711c874a5a734f13908704b0fbbb57dde37df7))
* re-attach stored node summaries by span after a slug algorithm change ([10c226b](https://github.com/GetActionSpace/mcp-context-server/commit/10c226b65f083a080867f5758abb043e5869993c))
* re-create the compressed payload table when compression is re-enabled ([6260302](https://github.com/GetActionSpace/mcp-context-server/commit/6260302dfbd31a88a62dd0066906a7c3a4939a0e))
* read numeric metadata-filter values as exact NUMERIC on PostgreSQL ([c1a3464](https://github.com/GetActionSpace/mcp-context-server/commit/c1a346439968b9049f786e2f3623463d66b930a5))
* refuse fp32 pgvector provisioning above the index dimension cap ([7fd2a6f](https://github.com/GetActionSpace/mcp-context-server/commit/7fd2a6f23dc9202c9aa4a8a5dce501cba50569de))
* refuse to disable compression while compressed data is present ([4a8c874](https://github.com/GetActionSpace/mcp-context-server/commit/4a8c874da80f1a1aee1131e5b7ef36f342a9361d))
* refuse to enable compression while uncompressed fp32 embeddings exist ([f04d065](https://github.com/GetActionSpace/mcp-context-server/commit/f04d0653f516cdb75850ec031ad67d59c26a6f0e))
* regenerate a reused summary on dedup divergence and honor empty replacements ([df567da](https://github.com/GetActionSpace/mcp-context-server/commit/df567da84c5b4a7cac532a26ff992fa859b0c737))
* regenerate reconcile summaries once per text and source pair ([65b7402](https://github.com/GetActionSpace/mcp-context-server/commit/65b740200ca8c252949a623358ab7e3deaa258eb))
* reject a blank DB_PATH at the configuration boundary ([62f0434](https://github.com/GetActionSpace/mcp-context-server/commit/62f0434dbb61216861ae5eb62255b9da11ab2822))
* reject a boolean for the ordered metadata comparison operators ([b3637f9](https://github.com/GetActionSpace/mcp-context-server/commit/b3637f918c69323748087131d895d93ad3a65307))
* reject a non-finite float in per-image metadata before generation ([4fad432](https://github.com/GetActionSpace/mcp-context-server/commit/4fad432e09c16e8893c3d1a2106871ad71026001))
* reject a non-string image data or mime_type in the shared validator ([943680a](https://github.com/GetActionSpace/mcp-context-server/commit/943680ab63f60e428e2321af68f89ef3bea1de78))
* reject a pool floor above the pool ceiling at the configuration boundary ([edd5c2b](https://github.com/GetActionSpace/mcp-context-server/commit/edd5c2b504cc2ae0c1d97d36a5821b57f2574e4d))
* reject a sub-minute UTC offset that SQLite evaluates to NULL ([596ca87](https://github.com/GetActionSpace/mcp-context-server/commit/596ca87f3def25f51d3e75da0a9b44f3dddf1ed2))
* reject a trailing-newline metadata key on both validators ([0ea56b5](https://github.com/GetActionSpace/mcp-context-server/commit/0ea56b5cc19b6c315fc31db88850602cd50481e5))
* reject a zero retry budget at the configuration boundary ([18f195c](https://github.com/GetActionSpace/mcp-context-server/commit/18f195c42fb2a18ce1ae6477c865fbc6b7abf4b0))
* reject an embedding dimension above the pgvector index cap ([71d11f3](https://github.com/GetActionSpace/mcp-context-server/commit/71d11f321b60b55b41322d66740c8e3e1f2332e9))
* reject an image mime_type that PostgreSQL cannot store ([177887f](https://github.com/GetActionSpace/mcp-context-server/commit/177887f9405dfe849d25b2b8c9eb36678eb580df))
* reject an image whose base64 data decodes to zero bytes ([1870dfb](https://github.com/GetActionSpace/mcp-context-server/commit/1870dfb630aed02cb87dc55ba8a2d33c3f78a5a5))
* reject non-string thread_id and text in the batch tools ([f459f57](https://github.com/GetActionSpace/mcp-context-server/commit/f459f57f4552c29078749a9e05d9b20db38abafd))
* reject NUL bytes in full-text queries and keep client input errors out of the circuit breaker ([97d75db](https://github.com/GetActionSpace/mcp-context-server/commit/97d75db5a124dd00f77d3965f21594ac6069063e))
* reject PostgreSQL-unstorable client input and exempt invalid metadata filters from the circuit breaker ([99673c7](https://github.com/GetActionSpace/mcp-context-server/commit/99673c7b70cde961ee7c0a00ec6bf6ad88d26d11))
* reject tag filters that normalize to empty ([7780476](https://github.com/GetActionSpace/mcp-context-server/commit/7780476f89eb5f47318c13eeebea2b913e92ab4b))
* reject unpaired-surrogate FTS queries via the shared bind probe ([d0b1fa6](https://github.com/GetActionSpace/mcp-context-server/commit/d0b1fa62c2e3f3521c6228c0eeda74c96b7ff4fa))
* relax image mime_type to a free-form advisory label ([1903b5f](https://github.com/GetActionSpace/mcp-context-server/commit/1903b5f2fd833f09db8b46d32f549cdd8b352522))
* render a single OPENAI_API_KEY in the Helm chart ([61b4a63](https://github.com/GetActionSpace/mcp-context-server/commit/61b4a63160a96153f44a99b77726771af77ecd8d))
* repair model-generated summaries that PostgreSQL cannot store ([85f6b6f](https://github.com/GetActionSpace/mcp-context-server/commit/85f6b6feafcf4944292b4cc347f6dd76c1330ff5))
* replace MAX(id) with array_agg in list_threads PostgreSQL branch ([8870c45](https://github.com/GetActionSpace/mcp-context-server/commit/8870c4530d443ed62d3da6dbfacd7b39880da27c))
* report not-found when an atomic batch update loses its row mid-write ([79bc785](https://github.com/GetActionSpace/mcp-context-server/commit/79bc78524fe92b2d2415582159bd6000f8a2eac9))
* report repeated metadata index fields with an accurate diagnostic ([c9ccc5a](https://github.com/GetActionSpace/mcp-context-server/commit/c9ccc5aae6bbac529fd40d049bfe14192a575b4c))
* require a positive image size limit and correct the hybrid toggle docs ([911a44a](https://github.com/GetActionSpace/mcp-context-server/commit/911a44a475ad0d7c99f46ae0ee33d5235021b38b))
* require positive connection-pool sizes and drop the unreachable embedding-dimension check ([4e3461c](https://github.com/GetActionSpace/mcp-context-server/commit/4e3461ca312148f3f89b9dfadc0e2034375d64d0))
* resolve an in-flight write future when the queue processor shuts down ([89c28ff](https://github.com/GetActionSpace/mcp-context-server/commit/89c28ff611d1047fd7cb3a7fff7990ccb838381c))
* resolve four code-audit defects in store/search/batch paths ([2d55a21](https://github.com/GetActionSpace/mcp-context-server/commit/2d55a21a231f73f983596712668375415026c311))
* resolve grep/navigate, index_tree, and breaker control-flow defects ([742d8d5](https://github.com/GetActionSpace/mcp-context-server/commit/742d8d5cf0b1022b7c07ba14953863314acfd749))
* resolve the full-text-search availability probe through search_path ([f9e8fa8](https://github.com/GetActionSpace/mcp-context-server/commit/f9e8fa8264ba3b74941c88e2b2f085840d36a722))
* resolve the SQLite target embedding dimension from settings in the migration CLI ([5810a3a](https://github.com/GetActionSpace/mcp-context-server/commit/5810a3a9612186f0b4aa1d588fb0aaf34ca27cbd))
* restrict metadata index field names to SQL identifiers ([80c2e5a](https://github.com/GetActionSpace/mcp-context-server/commit/80c2e5a66d0727081eee717a49c76a163267c3ea))
* retake the navigate snapshot when a concurrent update splits its reads ([41cd013](https://github.com/GetActionSpace/mcp-context-server/commit/41cd0135088b151a7c1d29bf9876cf2db9db4358))
* retarget pool-timeout advisory to connection acquire-wait ([8a727d5](https://github.com/GetActionSpace/mcp-context-server/commit/8a727d5f333e1bde46263c24e7f53f37c61ffca8))
* retry PostgreSQL writes cancelled by statement timeout ([d353f0d](https://github.com/GetActionSpace/mcp-context-server/commit/d353f0d34462d2e65ea8941b1501411f3c5ef32d))
* retry SQLite lock contention instead of charging the circuit breaker ([cb2eb0b](https://github.com/GetActionSpace/mcp-context-server/commit/cb2eb0bd1c15b886c85a4c83d74ada74dbcefb27))
* retry transaction-rollback failures instead of charging the circuit breaker ([23f5821](https://github.com/GetActionSpace/mcp-context-server/commit/23f582104c56c626d0541ad056ac6e89349ecad0))
* return structured validation errors when all hybrid search modes fail ([0b39333](https://github.com/GetActionSpace/mcp-context-server/commit/0b39333c2668e6d35faf629840ea4c6844f1c7f5))
* reverse response counters for batch entries discarded at commit time ([e20ccd7](https://github.com/GetActionSpace/mcp-context-server/commit/e20ccd7011088575b6000a9119631bf44ab85f46))
* run every PostgreSQL migration DDL under the configured migration timeout ([b5d334e](https://github.com/GetActionSpace/mcp-context-server/commit/b5d334ec912b524d12c89460f156dc7fb002a3e2))
* run metadata index DDL under the configured migration timeout ([6172899](https://github.com/GetActionSpace/mcp-context-server/commit/6172899aebc47b1184470f87e535730a30c9a856))
* run PostgreSQL migration DDL under the configured migration timeout ([b3cde8b](https://github.com/GetActionSpace/mcp-context-server/commit/b3cde8b4e84358832acc6fcb66e04a5fb05b4386))
* run schema-init statements under the migration timeout budget ([9375170](https://github.com/GetActionSpace/mcp-context-server/commit/93751709fa02541ba52a6d1e9b539faed67a21e5))
* run SQLite transaction closures on the executor thread pool ([1a970bb](https://github.com/GetActionSpace/mcp-context-server/commit/1a970bbbc81d65980d566722f746b9c08aebe9a6))
* run the compression CLI recounts under the migration budget ([093d081](https://github.com/GetActionSpace/mcp-context-server/commit/093d08116e0d51a140c3678ffb6c56d25a7df3e1))
* run the compression CLI's estimate and zero-data counts under the migration budget ([ae4d471](https://github.com/GetActionSpace/mcp-context-server/commit/ae4d471e3d0525afc2350ac2be0ed4551db7b8c9))
* run the compression migration DDL under the migration timeout ([3fa712e](https://github.com/GetActionSpace/mcp-context-server/commit/3fa712e1f1f5dc3886ffd2972ba3acfd121ba46a))
* run the full-text-search language migration under the configured migration timeout ([5de946a](https://github.com/GetActionSpace/mcp-context-server/commit/5de946aa25fdb4e151c2aa90b4a1d20170296092))
* run the full-text-search migration probes under the migration timeout ([d19091a](https://github.com/GetActionSpace/mcp-context-server/commit/d19091aadfc6af4ab593088d2f55bc7ee525681f))
* run the steady-state fingerprint ensure under the migration discipline ([c24a382](https://github.com/GetActionSpace/mcp-context-server/commit/c24a3829f3b5b789ad0164c923fa7f6e1750e613))
* screen grep tags for strings PostgreSQL cannot bind ([e67faae](https://github.com/GetActionSpace/mcp-context-server/commit/e67faae165315de592637fa265e79c3d9abd1e04))
* send Ollama summary calls with reasoning off ([#136](https://github.com/GetActionSpace/mcp-context-server/issues/136)) ([12ff258](https://github.com/GetActionSpace/mcp-context-server/commit/12ff258527ba56ff63f40874bc9d1939a4a6ec3d))
* serialize the BLAS thread-count pin across concurrent codec calls ([2698a6e](https://github.com/GetActionSpace/mcp-context-server/commit/2698a6e129ddddd4ec1e4d772a7cdf662644b9b9))
* serve every ranked-search page from one fixed-depth ordering ([e38d434](https://github.com/GetActionSpace/mcp-context-server/commit/e38d4347406f127d280f5ba46d09e7bf47e573b0))
* service a dequeued write before honoring a same-batch shutdown signal ([4eec7f0](https://github.com/GetActionSpace/mcp-context-server/commit/4eec7f0ff9cc58b48d8789a5cfd8ee1b585d0f8b))
* skip pgvector provisioning for a compressed generation-on server ([00213f8](https://github.com/GetActionSpace/mcp-context-server/commit/00213f8e9cf70f827bd612203ca5388fcf2cfe92))
* skip PostgreSQL-unstorable rows in the SQLite-to-PostgreSQL migration instead of aborting ([9aa2103](https://github.com/GetActionSpace/mcp-context-server/commit/9aa21037ba2cc1ee85f5f46b5ccda5f9f13cd0fb))
* skip rows with malformed metadata JSON in the migration pre-check ([c03d09e](https://github.com/GetActionSpace/mcp-context-server/commit/c03d09ee83fa0feb11c10f2de1c62c14eef2d275))
* snapshot the duplicate candidate summary atomically in the pre-check ([b887f53](https://github.com/GetActionSpace/mcp-context-server/commit/b887f53e044423e3df96aa672f6ef93f3b7e30b6))
* stop case-sensitive string array_contains matching container elements on SQLite ([42ff7c5](https://github.com/GetActionSpace/mcp-context-server/commit/42ff7c559842f88bc16b71a9a9323a6ec3318dd0))
* stop charging the circuit breaker for transient PostgreSQL write conditions ([31c2c84](https://github.com/GetActionSpace/mcp-context-server/commit/31c2c8435614ef2d6f7a644a063abd7fd9e0b662))
* stop retrying pool-saturation timeouts at the tool layer ([5d70fcb](https://github.com/GetActionSpace/mcp-context-server/commit/5d70fcb6694cfa056758237f58cd1a7883812de1))
* strip internal chunk-boundary fields from semantic search results ([b5e2ee7](https://github.com/GetActionSpace/mcp-context-server/commit/b5e2ee7fecc7dfbc53c6d0bcbd1ba418df868713))
* surface a clean not-found error when an update targets a deleted entry ([8b34955](https://github.com/GetActionSpace/mcp-context-server/commit/8b3495558847e11545c326b60661554cfcd19096))
* thread the pgvector decision through the decompress CLI so it runs on a host without pgvector ([d358e59](https://github.com/GetActionSpace/mcp-context-server/commit/d358e595433f24f3d563b443102b831f20d738bd))
* tolerate a compression_metadata table without the fingerprint column ([0878ee5](https://github.com/GetActionSpace/mcp-context-server/commit/0878ee5233848af8901ef97ad099597881e6a1d3))
* tolerate a source embedding_chunks table without the boundary columns ([4d3868b](https://github.com/GetActionSpace/mcp-context-server/commit/4d3868bf67ed2ec2c893898457a58d2dc13347d6))
* type PostgreSQL connection-establishment timeouts and charge acquire-phase faults ([8fd0b52](https://github.com/GetActionSpace/mcp-context-server/commit/8fd0b52891eb683fd7a36539acfbf4cb52a5d67a))
* type the returned image shape for context and search results ([99debc6](https://github.com/GetActionSpace/mcp-context-server/commit/99debc6d1cac0912d4808fc7a48c96242c87eb37))
* unify ID-prefix resolution and align PostgreSQL is_null semantics ([6fdc997](https://github.com/GetActionSpace/mcp-context-server/commit/6fdc997d863c2223b3c10f4c18751b4b9c827a5e))
* unwedge and guard the zero-data decompress path ([ed6c353](https://github.com/GetActionSpace/mcp-context-server/commit/ed6c353e7c4dafa9487579f500bebac0bfff8af5))
* upgrade FastMCP to 4.1.1 and the pinned dependencies ([#133](https://github.com/GetActionSpace/mcp-context-server/issues/133)) ([7aee523](https://github.com/GetActionSpace/mcp-context-server/commit/7aee523784023e0902e8ff1ad9c4842cf110d45b))
* upgrade FastMCP to 4.1.3 and the LangChain packages ([#135](https://github.com/GetActionSpace/mcp-context-server/issues/135)) ([3bf7b21](https://github.com/GetActionSpace/mcp-context-server/commit/3bf7b21c8ad70c54f9159b3c2aa0208cfd67ad9d))
* URL-encode the user and database in the PostgreSQL connection string ([714d3e6](https://github.com/GetActionSpace/mcp-context-server/commit/714d3e685128d8bec2a5b508cb8059d1442dd208))
* validate an omitted metadata-filter value against its operator ([8f15976](https://github.com/GetActionSpace/mcp-context-server/commit/8f15976835f18c1e7c97b778523345c97a41249b))
* validate stored node-id hits against their recorded span ([c758202](https://github.com/GetActionSpace/mcp-context-server/commit/c758202fdbf07851d99e674f3b55e2a5eda364bd))
* validate the settings-fallback embedding dimension in the migrate pre-flight ([dcf87b9](https://github.com/GetActionSpace/mcp-context-server/commit/dcf87b92e4bb6ea4780db5d13851ba6291e0733a))
* validate typed metadata and image payloads before generation runs ([3afeeb3](https://github.com/GetActionSpace/mcp-context-server/commit/3afeeb3858f4cd0aafda15838ee9a72548a901fa))
* verify the codebook fingerprint before decompressing embeddings ([9221304](https://github.com/GetActionSpace/mcp-context-server/commit/92213045387a96a180ba127e454d02d4855386fc))
* verify the realized codebook fingerprint before CLI embedding backfill ([fcd761a](https://github.com/GetActionSpace/mcp-context-server/commit/fcd761ae1f4961d7d6599944a9649bc79ad1989b))

## [2.2.2](https://github.com/alex-feel/mcp-context-server/compare/v2.2.1...v2.2.2) (2026-03-28)


### Bug Fixes

* add interleaving check to deduplication to preserve conversational turn ordering ([5ef8b2f](https://github.com/alex-feel/mcp-context-server/commit/5ef8b2fae6cf1703722de2bb27de3840791d495d))
* bump pyjwt and orjson to fix HIGH-severity CVEs ([9489e85](https://github.com/alex-feel/mcp-context-server/commit/9489e8543c4fb92a394d5aacf39e9f2d2cdf998c))
* extract shared tool infrastructure and resolve batch-nonbatch parity bugs ([f337a67](https://github.com/alex-feel/mcp-context-server/commit/f337a67a82e796d8055a588878fdf20c49fe1dc1))
* pass API keys explicitly in summary providers and clean up project config ([ff03172](https://github.com/alex-feel/mcp-context-server/commit/ff031722896e97846eddfb0bcf54cbaa84af8cc7))

## [2.2.1](https://github.com/alex-feel/mcp-context-server/compare/v2.2.0...v2.2.1) (2026-03-26)


### Bug Fixes

* resolve hybrid search quoted phrase loss, embedding guard asymmetry, and batch content_type drift ([069599b](https://github.com/alex-feel/mcp-context-server/commit/069599bf046a42bdd149fcb6d954a898be3f4c5a))
* serialize SQLite write queue operations through writer lock ([10db4cc](https://github.com/alex-feel/mcp-context-server/commit/10db4cc2422e8a2b11b6fcb95fb2c242f9da9297))

## [2.2.0](https://github.com/alex-feel/mcp-context-server/compare/v2.1.0...v2.2.0) (2026-03-25)


### Features

* add site-packages path trimming to universal logger ([b5088ff](https://github.com/alex-feel/mcp-context-server/commit/b5088ff958ddc112507c42c940e49574326ac40e))
* update default OpenAI summary model to GPT-5.4 Nano ([1dfa0cd](https://github.com/alex-feel/mcp-context-server/commit/1dfa0cdd447ad7067817c188db1282f7d4a47c02))


### Bug Fixes

* prevent reasoning models from exhausting summary output token budget ([347cfc0](https://github.com/alex-feel/mcp-context-server/commit/347cfc02757dacb288be36d3a336821487d587e5))

## [2.1.0](https://github.com/alex-feel/mcp-context-server/compare/v2.0.0...v2.1.0) (2026-03-18)


### Features

* add automatic model pulling for Ollama providers on startup ([3196c2b](https://github.com/alex-feel/mcp-context-server/commit/3196c2bc2eb236908202ce44dffbdae7913f3ccc))
* add context window validation for summary generation ([7ea909e](https://github.com/alex-feel/mcp-context-server/commit/7ea909ed68a2203022315754bc30aaac8b821fa7))
* add eager model loading and concurrency protection for FlashRank reranking ([f260d27](https://github.com/alex-feel/mcp-context-server/commit/f260d272fb63392350abfb76f856eb771ae8a9ea))
* add Ollama cold-start optimization with model pre-warming ([41e2776](https://github.com/alex-feel/mcp-context-server/commit/41e2776d8d40963d38e2cf339325da822655659d))
* add SHA-256 content hash for deduplication optimization ([76ec45f](https://github.com/alex-feel/mcp-context-server/commit/76ec45f24d805c3055f4b191131528c013a966b8))
* add Skill Integration section to DEFAULT_INSTRUCTIONS ([7b84ae0](https://github.com/alex-feel/mcp-context-server/commit/7b84ae07cac3c72c4c6200a0d79379b5a70bfd23))
* change default summary model to qwen3:0.6b ([c5c7c0e](https://github.com/alex-feel/mcp-context-server/commit/c5c7c0e4ee63a58fd8aee1ead2c6f15f728cb44a))


### Bug Fixes

* classify PostgreSQL initialization errors and limit Docker restart retries ([ef6e58a](https://github.com/alex-feel/mcp-context-server/commit/ef6e58ad10327e8f4cd63318646f7d8542d48548))
* eliminate redundant PostgreSQL schema execution ([976d625](https://github.com/alex-feel/mcp-context-server/commit/976d625d3274e88a13f6c5ed252710cbc5c4673b))
* enforce uniform Generation-First Transactional Integrity across all tools ([a1bb652](https://github.com/alex-feel/mcp-context-server/commit/a1bb652795f76012893cfad8fb82d627868695d1))
* replace per-call-site JSON string deserialization with schema-aware FastMCP middleware ([f706b5b](https://github.com/alex-feel/mcp-context-server/commit/f706b5bae9e9d6b1789276b1cb01682740cd0949))
* replace provider-availability checks with actual generation counters in batch tool messages ([a473919](https://github.com/alex-feel/mcp-context-server/commit/a4739190f3d81662675f11383693d3ab1a20d625))
* resolve &lt;unknown&gt; in tenacity retry log messages ([91e4bb4](https://github.com/alex-feel/mcp-context-server/commit/91e4bb4bbb4f9548053b96fc9f2121838fc604bb))
* resolve Docker and Helm deployment issues with model pull and summary support ([1c893d1](https://github.com/alex-feel/mcp-context-server/commit/1c893d165c57c8c7467539051e852590fa5af9e8))
* use source-aware dynamic prompts for summary generation ([5c80bea](https://github.com/alex-feel/mcp-context-server/commit/5c80bea54497d45c812b0a821ab01f40527089a2))

## [2.0.0](https://github.com/alex-feel/mcp-context-server/compare/v1.7.1...v2.0.0) (2026-03-13)


### ⚠ BREAKING CHANGES

* This release introduces two breaking changes:

### Features

* add LLM-powered summary generation for context entries ([9f4c9e2](https://github.com/alex-feel/mcp-context-server/commit/9f4c9e2706beedbc25946365ce00d73eb9c6a5b9))

## [1.7.1](https://github.com/alex-feel/mcp-context-server/compare/v1.7.0...v1.7.1) (2026-03-09)


### Bug Fixes

* change FASTMCP_STATELESS_HTTP default from false to true ([277baa3](https://github.com/alex-feel/mcp-context-server/commit/277baa3b8cacf82f93d0f5a47abe0857a73c6bcd))
* prevent event loop blocking in FlashRank model loading ([b0076fd](https://github.com/alex-feel/mcp-context-server/commit/b0076fdabc23622d50d6a1b8d243daf2cfc23054))

## [1.7.0](https://github.com/alex-feel/mcp-context-server/compare/v1.6.0...v1.7.0) (2026-03-08)


### Features

* add configurable micro-batching for FlashRank reranking ([43fcd9c](https://github.com/alex-feel/mcp-context-server/commit/43fcd9c12a4654127b756dd89fdb565500c08304))


### Bug Fixes

* apply ts_headline only to LIMIT'd rows in PostgreSQL FTS ([2cdb52e](https://github.com/alex-feel/mcp-context-server/commit/2cdb52e808cef57a35226e6bcdf5bae957a0799a))
* improve tool description clarity and self-sufficiency ([ed522c3](https://github.com/alex-feel/mcp-context-server/commit/ed522c3754a46a7665cfab4eb23ffd3ec470c4af))
* offload FlashRank ONNX inference to thread pool via asyncio.to_thread ([272eb9e](https://github.com/alex-feel/mcp-context-server/commit/272eb9e8b605e837a743c0697b0f818091ecf547))
* report project version in MCP protocol handshake ([0372f58](https://github.com/alex-feel/mcp-context-server/commit/0372f58f4d1e5a2a6cdeb5c1d05762225e9f6a9c))
* skip auth initialization on stdio transport ([7fbc9ef](https://github.com/alex-feel/mcp-context-server/commit/7fbc9ef8ba1a7572cba1d79ae820520ee9000c46))

## [1.6.0](https://github.com/alex-feel/mcp-context-server/compare/v1.5.0...v1.6.0) (2026-03-04)


### Features

* add adaptive FTS mode to hybrid search for improved long-query recall ([54c5f31](https://github.com/alex-feel/mcp-context-server/commit/54c5f31b1d1c875ef72c6b647096a802190d887b))
* add RERANKING_CPU_MEM_ARENA env var to control ONNX memory arena ([5f98cf7](https://github.com/alex-feel/mcp-context-server/commit/5f98cf7e79af9ae321ac3cdce298bb73b4f4f711))


### Bug Fixes

* add temporary monkey-patches for MCP SDK session crash on client disconnect ([d9a9f3d](https://github.com/alex-feel/mcp-context-server/commit/d9a9f3d21ae66b98ae61100a394a6485d0c4f5a3))
* correct deduplication data integrity in store_context and store_context_batch ([a395ede](https://github.com/alex-feel/mcp-context-server/commit/a395ede0378b1342d774846d4adaae11f57f9944))
* correct modes_used to reflect execution rather than results in hybrid search ([c747edf](https://github.com/alex-feel/mcp-context-server/commit/c747edfd46ccbd933285c68aac4d7879249527e0))

## [1.5.0](https://github.com/alex-feel/mcp-context-server/compare/v1.4.0...v1.5.0) (2026-03-02)


### Features

* add configurable ONNX intra-op thread control for FlashRank reranking ([1352988](https://github.com/alex-feel/mcp-context-server/commit/1352988939b92e5c1cb2ad251a09f5e17f8b98e9))
* add embedding concurrency control, dynamic timeout, and search quality improvements ([757d19f](https://github.com/alex-feel/mcp-context-server/commit/757d19fdbf658b1990a039e45e27a2d7ba1e546d))


### Bug Fixes

* sanitize hybrid search warning messages to avoid leaking error details ([d71b0c9](https://github.com/alex-feel/mcp-context-server/commit/d71b0c977af29735008e5d9e8b20149eaa12c615))
* suppress onnxruntime type errors via mypy and pyright config ([e59e9de](https://github.com/alex-feel/mcp-context-server/commit/e59e9de46b101af2516ee0f22905ccb018b93c9c))

## [1.4.0](https://github.com/alex-feel/mcp-context-server/compare/v1.3.3...v1.4.0) (2026-02-27)


### Features

* add FASTMCP_STATELESS_HTTP setting for horizontal scaling ([c94a423](https://github.com/alex-feel/mcp-context-server/commit/c94a42357b735eb417f92c57c3a2851e535cac8e))
* add MCP server instructions support ([e1805a6](https://github.com/alex-feel/mcp-context-server/commit/e1805a6dfb3c1344d596547c9c0af3af8e124fc1))
* remove search_modes parameter from hybrid_search_context ([2335d0c](https://github.com/alex-feel/mcp-context-server/commit/2335d0cc6ab5bd72e5cd76983cef87d385f7cc79))


### Bug Fixes

* remove install hints from inner exceptions in providers and services ([e9708b2](https://github.com/alex-feel/mcp-context-server/commit/e9708b2245097f5d0623bae11cee3210efee145a))
* replace session-scoped advisory locks with transaction-scoped locks in migrations ([4508eb8](https://github.com/alex-feel/mcp-context-server/commit/4508eb81918eca61ce40f5141d79119a3adf6c97))
* update dependencies to resolve Trivy CVE findings ([c21d736](https://github.com/alex-feel/mcp-context-server/commit/c21d7362bb8f0e1e719c58bb984a58a316e099bd))

## [1.3.3](https://github.com/alex-feel/mcp-context-server/compare/v1.3.2...v1.3.3) (2026-02-11)


### Bug Fixes

* add advisory locks to PostgreSQL migrations and schema initialization ([d637cc5](https://github.com/alex-feel/mcp-context-server/commit/d637cc547d32fa5ab565930abe57321f68aa3ec9))
* add PostgreSQL connection resilience defensive hardening ([e0ae6da](https://github.com/alex-feel/mcp-context-server/commit/e0ae6da0becdea7f93da63596287c116fa8f3131))
* add ROLLBACK to PostgreSQL backend ([bb69f45](https://github.com/alex-feel/mcp-context-server/commit/bb69f45dd0129dfcbb8c48d57e60571ab2e3c1d8))

## [1.3.2](https://github.com/alex-feel/mcp-context-server/compare/v1.3.1...v1.3.2) (2026-02-06)


### Bug Fixes

* add python-multipart constraint for CVE-2026-24486 ([9c6d2ea](https://github.com/alex-feel/mcp-context-server/commit/9c6d2ea7d39f7f2f579559ccdbc32c42031afa8f))
* resolve embedding deduplication race condition ([89f6596](https://github.com/alex-feel/mcp-context-server/commit/89f6596d18bbc10e18662ed86f62c46fe80fd5a1))
* resolve PostgreSQL migration timeout and idempotency issues ([d2f8f8e](https://github.com/alex-feel/mcp-context-server/commit/d2f8f8e3787c891ef6bde67652ec9fb3c8522406))

## [1.3.1](https://github.com/alex-feel/mcp-context-server/compare/v1.3.0...v1.3.1) (2026-01-25)


### Bug Fixes

* classify PostgreSQL backend errors to control container restart behavior ([f5ed4c8](https://github.com/alex-feel/mcp-context-server/commit/f5ed4c8ac0cec0d7ead6d85d018dcb7dc44bc1ea))
* upgrade FastMCP to 2.14.4 ([8780345](https://github.com/alex-feel/mcp-context-server/commit/8780345f35fee558af266f9d33c226592ccdf9e0))

## [1.3.0](https://github.com/alex-feel/mcp-context-server/compare/v1.2.1...v1.3.0) (2026-01-20)


### Features

* add Pgpool-II detection for PostgreSQL backend ([e64f082](https://github.com/alex-feel/mcp-context-server/commit/e64f0826238417bfd5624de10d94dd31b3da83cf))

## [1.2.1](https://github.com/alex-feel/mcp-context-server/compare/v1.2.0...v1.2.1) (2026-01-18)


### Bug Fixes

* add backend-specific FTS tool descriptions ([09113cd](https://github.com/alex-feel/mcp-context-server/commit/09113cd9fef0e5aec83c9dbc73be2b77f3b311e0))

## [1.2.0](https://github.com/alex-feel/mcp-context-server/compare/v1.1.0...v1.2.0) (2026-01-17)


### Features

* add configurable asyncpg prepared statement cache settings ([5ebf287](https://github.com/alex-feel/mcp-context-server/commit/5ebf2876bbee7c089f7b3901a54c9eebb8fac281))

## [1.1.0](https://github.com/alex-feel/mcp-context-server/compare/v1.0.0...v1.1.0) (2026-01-17)


### Features

* implement embedding-first transactional integrity ([c9e4c12](https://github.com/alex-feel/mcp-context-server/commit/c9e4c12957018510732a7bf7815c2f98ebe9e88b))

## [1.0.0](https://github.com/alex-feel/mcp-context-server/compare/v0.17.0...v1.0.0) (2026-01-16)


### ⚠ BREAKING CHANGES

* API response structure changed for all search tools. Users must update their code:
    - FTS: result['score'] -> result['scores']['fts_score']
    - Semantic: result['distance'] -> result['scores']['semantic_distance']
    - All tools: result['rerank_score'] -> result['scores']['rerank_score']
* Default embedding model changed from embeddinggemma:latest (768 dim) to qwen3-embedding:0.6b (1024 dim). Existing vector databases will have incompatible embeddings.
* ENABLE_EMBEDDING_GENERATION now defaults to true. Server will NOT start if embedding dependencies are not available when ENABLE_EMBEDDING_GENERATION=true (the default).

### Features

* add chunk-aware reranking for FTS, semantic search, and hybrid search ([ff79859](https://github.com/alex-feel/mcp-context-server/commit/ff79859a65645be5fe83b50054994ee1ac352f48))
* add embedding truncation control with universal validator ([d941b6f](https://github.com/alex-feel/mcp-context-server/commit/d941b6f349ddac787abfee581e5f22e6b8103579))
* add text chunking and cross-encoder reranking ([942bb24](https://github.com/alex-feel/mcp-context-server/commit/942bb24d66330017864d91d1a73d07f9df7920ed))
* implement universal retry wrapper for embedding providers ([4031061](https://github.com/alex-feel/mcp-context-server/commit/40310619dfdf09be9ef1ca24bebbff8160ac5a92))
* prevent Docker infinite restart loops with exit code handling ([91acfa4](https://github.com/alex-feel/mcp-context-server/commit/91acfa4c00cd979577ee2157619a7b495caff8d7))
* replace embeddinggemma with qwen3-embedding:0.6b ([02ce75e](https://github.com/alex-feel/mcp-context-server/commit/02ce75e3833eb7150e873f33e670c8809d26da6c))
* separate embedding generation from semantic search ([3967262](https://github.com/alex-feel/mcp-context-server/commit/396726205d2a681e939d1a0fe36ba96c2822b7aa))
* standardize scores API across all search tools ([0a940c9](https://github.com/alex-feel/mcp-context-server/commit/0a940c9d134b44a8e7d1c3e4fca637aad0b1566a))
* switch PostgreSQL FTS ranking from ts_rank to ts_rank_cd ([a58fd9c](https://github.com/alex-feel/mcp-context-server/commit/a58fd9c9e4d24e91cc26eb53b31732fa4aa4a845))


### Bug Fixes

* enable LangSmith tracing for embedding operations ([ba7bf9f](https://github.com/alex-feel/mcp-context-server/commit/ba7bf9fc12a66e673dac54d9d947970b5f380587))

## [0.17.0](https://github.com/alex-feel/mcp-context-server/compare/v0.16.1...v0.17.0) (2026-01-11)


### Features

* implement LangChain embeddings multi-provider architecture ([7382629](https://github.com/alex-feel/mcp-context-server/commit/7382629a74da6fb39d5e635096d7eea1c619e3c8))


### Bug Fixes

* add schema qualification to recursive jsonb_merge_patch call ([51bc48c](https://github.com/alex-feel/mcp-context-server/commit/51bc48c8ba86e4ae990b7bb3ef5efa15c8e1a6b7))
* resolve critical PostgreSQL backend issues ([8186f90](https://github.com/alex-feel/mcp-context-server/commit/8186f90da0804f5adf858cf857d25e8b81468649))

## [0.16.1](https://github.com/alex-feel/mcp-context-server/compare/v0.16.0...v0.16.1) (2026-01-09)


### Bug Fixes

* add POSTGRESQL_SCHEMA setting and refactor metadata index management ([772f3aa](https://github.com/alex-feel/mcp-context-server/commit/772f3aa6c7d31945e69fd3ed31ddc69ae7da1454))
* improve error handling and add timeout/retry logic ([319e08f](https://github.com/alex-feel/mcp-context-server/commit/319e08fdca103e2e835c9318984c10e931db6e75))
* replace hardcoded public schema with POSTGRESQL_SCHEMA setting ([3e6511f](https://github.com/alex-feel/mcp-context-server/commit/3e6511fc927ff9c0c7c55480c4841cd6b31eddfb))

## [0.16.0](https://github.com/alex-feel/mcp-context-server/compare/v0.15.1...v0.16.0) (2026-01-06)


### Features

* add configurable metadata field indexing with sync modes ([10ab4cb](https://github.com/alex-feel/mcp-context-server/commit/10ab4cb5924ae7230c146e39a8786a755b3f44fa))

## [0.15.1](https://github.com/alex-feel/mcp-context-server/compare/v0.15.0...v0.15.1) (2026-01-05)


### Bug Fixes

* add array_contains to MCP tool metadata_filters descriptions ([45f2357](https://github.com/alex-feel/mcp-context-server/commit/45f23572fe6250c570f0d41f272996311f617857))

## [0.15.0](https://github.com/alex-feel/mcp-context-server/compare/v0.14.1...v0.15.0) (2026-01-05)


### Features

* add array_contains operator with graceful non-array handling ([b23642a](https://github.com/alex-feel/mcp-context-server/commit/b23642a2933ac267fbbb9519062a252bef64d8c8))

## [0.14.1](https://github.com/alex-feel/mcp-context-server/compare/v0.14.0...v0.14.1) (2025-12-30)


### Bug Fixes

* add search_path to functions for CVE-2018-1058 mitigation ([d156235](https://github.com/alex-feel/mcp-context-server/commit/d156235547f0ab5b5837022bb91bbdfdba94edf9))

## [0.14.0](https://github.com/alex-feel/mcp-context-server/compare/v0.13.0...v0.14.0) (2025-12-28)


### Features

* add HTTP authentication support with bearer token and OAuth options ([e625b42](https://github.com/alex-feel/mcp-context-server/commit/e625b42add2360f1bd9212690182b0b8474b03f8))

## [0.13.0](https://github.com/alex-feel/mcp-context-server/compare/v0.12.0...v0.13.0) (2025-12-27)


### Features

* add DISABLED_TOOLS environment variable and MCP tool annotations ([f6bccd7](https://github.com/alex-feel/mcp-context-server/commit/f6bccd7593c0bc2e9220299b5a9325ed76323850))

## [0.12.0](https://github.com/alex-feel/mcp-context-server/compare/v0.11.0...v0.12.0) (2025-12-26)


### Features

* add compose configuration for external PostgreSQL ([61b294f](https://github.com/alex-feel/mcp-context-server/commit/61b294f5c3112ffd46d4472146cac317f09aa245))
* add Docker deployment with HTTP transport support ([77390e9](https://github.com/alex-feel/mcp-context-server/commit/77390e9b552cd7b76d31bd0c89cccd699e65bb68))
* split compose files for independent SQLite and PostgreSQL deployments ([7840291](https://github.com/alex-feel/mcp-context-server/commit/784029108dd798a0dde86ba417dd8007876cda6b))

## [0.11.0](https://github.com/alex-feel/mcp-context-server/compare/v0.10.0...v0.11.0) (2025-12-22)


### Features

* standardize search tools API response structure ([04d7472](https://github.com/alex-feel/mcp-context-server/commit/04d7472dec0cb23e292b5405412c9aec0029f9d6))


### Bug Fixes

* add explain_query support to semantic search for consistent stats structure ([cd5884b](https://github.com/alex-feel/mcp-context-server/commit/cd5884b490f43df22150fa338a786c2af957ec22))
* remove duplicate Args sections from MCP tool docstrings ([9e0df55](https://github.com/alex-feel/mcp-context-server/commit/9e0df55b71ffd33154c30ba829030298730d7169))
* replace blocking pathlib.Path with anyio.Path in async code ([0101b0a](https://github.com/alex-feel/mcp-context-server/commit/0101b0a32566a790a89b62ba5ff42b3c6770713d))
* standardize MCP tool docstrings to use inline JSON structures ([7bef90d](https://github.com/alex-feel/mcp-context-server/commit/7bef90d55c2c08dcc79b94c2b2d2b20b80871cb7))

## [0.10.0](https://github.com/alex-feel/mcp-context-server/compare/v0.9.0...v0.10.0) (2025-12-21)


### Features

* add explain_query parameter to fts_search_context ([7334e9f](https://github.com/alex-feel/mcp-context-server/commit/7334e9fef5ac88f7806cc5ed9754a219fe703d5c))
* add explain_query parameter to hybrid_search_context ([b32129d](https://github.com/alex-feel/mcp-context-server/commit/b32129d6a8b60a45575aefc69a5536044ab2cd09))
* add hybrid search with RRF fusion ([8c6ea15](https://github.com/alex-feel/mcp-context-server/commit/8c6ea15ae7867880183a9ac93e855e586069838f))
* add offset, content_type, include_images to search tools ([bb22012](https://github.com/alex-feel/mcp-context-server/commit/bb2201257c20e4c645cc3f70863d8e8de133f55e))
* add tags parameter to semantic, FTS, and hybrid search ([bcf1332](https://github.com/alex-feel/mcp-context-server/commit/bcf1332f5015fae69be640e062f45fd60534fffd))
* add uniform backend display to all statistics output ([0d7c988](https://github.com/alex-feel/mcp-context-server/commit/0d7c988a25693bdced56cc89cb6235dda5f5d2f0))
* rename top_k to limit in semantic_search_context ([2c27c14](https://github.com/alex-feel/mcp-context-server/commit/2c27c1418e0055fdf835055c1efb87b37d3e4b0a))
* standardize limit parameter across all search tools ([d85d07d](https://github.com/alex-feel/mcp-context-server/commit/d85d07de4aada18d4c90375f4e17589692dc4d82))


### Bug Fixes

* handle hyphenated words in FTS queries ([b2f9ccb](https://github.com/alex-feel/mcp-context-server/commit/b2f9ccbf94ff1fa5ca875843504edff593de0d2c))
* hybrid search pagination and test quality improvements ([2da98fe](https://github.com/alex-feel/mcp-context-server/commit/2da98fe5a31d594832916ef6a968245d821fb6a1))
* make startup checks backend-specific ([3ee1149](https://github.com/alex-feel/mcp-context-server/commit/3ee1149638acdc8869ef8755aadbcb5f14dc9e04))
* register hybrid_search_context tool dynamically ([0ee0d22](https://github.com/alex-feel/mcp-context-server/commit/0ee0d22f9d1743eb51f5516facad6dcd74a7abae))

## [0.9.0](https://github.com/alex-feel/mcp-context-server/compare/v0.8.0...v0.9.0) (2025-12-03)


### Features

* add full-text search with FTS5 and PostgreSQL tsvector ([34ef13f](https://github.com/alex-feel/mcp-context-server/commit/34ef13f5c39f82dae84d41784dfc1d055dd63fa4))


### Bug Fixes

* handle boolean metadata filtering for PostgreSQL JSONB ([1ff407e](https://github.com/alex-feel/mcp-context-server/commit/1ff407e6891e3aec579aef2c531ac33f3e2b2452))

## [0.8.0](https://github.com/alex-feel/mcp-context-server/compare/v0.7.0...v0.8.0) (2025-11-30)


### Features

* add bulk operations for batch context management ([7516a93](https://github.com/alex-feel/mcp-context-server/commit/7516a935cef9fbe7bed5d7bc55dbf0001b41b830))
* add metadata filtering to semantic_search_context ([50346d6](https://github.com/alex-feel/mcp-context-server/commit/50346d64dccc067925e405b9ebbfa24dee94f459))


### Bug Fixes

* handle integer arrays in in/not_in metadata operators ([23076df](https://github.com/alex-feel/mcp-context-server/commit/23076df01b778464a1c4189e698d0782aacdfc28))

## [0.7.0](https://github.com/alex-feel/mcp-context-server/compare/v0.6.0...v0.7.0) (2025-11-29)


### Features

* support date filtering in search_context and semantic_search_context ([e433f5e](https://github.com/alex-feel/mcp-context-server/commit/e433f5e195b8bd3278f55ef4685d7be0844b3c2b))

## [0.6.0](https://github.com/alex-feel/mcp-context-server/compare/v0.5.1...v0.6.0) (2025-11-29)


### Features

* add metadata_patch parameter for partial metadata updates ([42f7a5f](https://github.com/alex-feel/mcp-context-server/commit/42f7a5fc33cd0b544db97b5de71406bbe78c8beb))

## [0.5.1](https://github.com/alex-feel/mcp-context-server/compare/v0.5.0...v0.5.1) (2025-11-26)


### Bug Fixes

* resolve Supabase and PostgreSQL issues ([3d5c450](https://github.com/alex-feel/mcp-context-server/commit/3d5c450b19311fbf2e25cc73339b5b9f2dcb2d4e))

## [0.5.0](https://github.com/alex-feel/mcp-context-server/compare/v0.4.1...v0.5.0) (2025-11-23)


### Features

* add PostgreSQL backend with pgvector semantic search ([03ad9b0](https://github.com/alex-feel/mcp-context-server/commit/03ad9b023ead730805bc448cf3d3ee7a7ea0f58f))
* add storage backend abstraction for multi-database support ([4e7744a](https://github.com/alex-feel/mcp-context-server/commit/4e7744a6924be6c1f4cd0448b54d944e7c6d6848))


### Bug Fixes

* eliminate duplicate tool registration and improve naming ([d46cb7c](https://github.com/alex-feel/mcp-context-server/commit/d46cb7c3b524a3609b9f78801a01c09f27429f79))
* eliminate redundant backend initializations during startup ([dcd14d7](https://github.com/alex-feel/mcp-context-server/commit/dcd14d73c2ab3b8854edb7e269aac5f58577470a))
* load sqlite-vec extension before semantic search migration ([3069460](https://github.com/alex-feel/mcp-context-server/commit/306946028d2a6a6ae86a1d6d8368036ebb19d2c9))
* resolve asyncio primitives event loop binding issue ([705fd9d](https://github.com/alex-feel/mcp-context-server/commit/705fd9d37af693298d482b787f7feb46b79beafe))
* resolve integration test hang with persistent backend ([35219db](https://github.com/alex-feel/mcp-context-server/commit/35219dbc79dc2d8bc823d00032773ccc02e2c345))

## [0.4.1](https://github.com/alex-feel/mcp-context-server/compare/v0.4.0...v0.4.1) (2025-10-10)


### Bug Fixes

* allow nested JSON structures in metadata ([7f624ee](https://github.com/alex-feel/mcp-context-server/commit/7f624ee82dd3ad6292f583bf04e0cd815d6e1ecf))

## [0.4.0](https://github.com/alex-feel/mcp-context-server/compare/v0.3.0...v0.4.0) (2025-10-06)


### Features

* add semantic search with EmbeddingGemma and sqlite-vec ([2e0d3db](https://github.com/alex-feel/mcp-context-server/commit/2e0d3db3616da98e8da418f7391c452a722aa3fa))
* enable configurable embedding dimensions for Ollama models ([2d68963](https://github.com/alex-feel/mcp-context-server/commit/2d68963de47d10c54442c8454e855792f388deae))


### Bug Fixes

* correct semantic search filtering with CTE-based pre-filtering ([66161a3](https://github.com/alex-feel/mcp-context-server/commit/66161a357b1a06e51058939135611063a4c1123f))
* resolve type checking errors for optional dependencies ([be47f9d](https://github.com/alex-feel/mcp-context-server/commit/be47f9d7f33b5634f68e922a05e60154af881091))

## [0.3.0](https://github.com/alex-feel/mcp-context-server/compare/v0.2.0...v0.3.0) (2025-10-04)


### Features

* add update_context tool for modifying existing context entries ([08aed11](https://github.com/alex-feel/mcp-context-server/commit/08aed11af11e4d1e476181a7885e7d90e7ad08a0))


### Bug Fixes

* enforce Pydantic validation and resolve test reliability issues ([6137efc](https://github.com/alex-feel/mcp-context-server/commit/6137efc83af36cf162a941836ede78811d68530b))
* ensure consistent validation patterns across all MCP tools ([7137aca](https://github.com/alex-feel/mcp-context-server/commit/7137acade3f6d1b7af1f98461ddab8cd80bb1e4e))
* move validation to Pydantic models ([1e2e480](https://github.com/alex-feel/mcp-context-server/commit/1e2e4803e94dc7d3e7f0c9965dfcfc3f727af17c))
* resolve all pre-commit issues and test failures ([0a2142d](https://github.com/alex-feel/mcp-context-server/commit/0a2142dc8a3d16b967693038982825af277ae82b))

## [0.2.0](https://github.com/alex-feel/mcp-context-server/compare/v0.1.0...v0.2.0) (2025-09-28)


### Features

* add comprehensive metadata filtering to search_context ([e22cfe0](https://github.com/alex-feel/mcp-context-server/commit/e22cfe0fac6294725d423823bdc2d5ff802f88f5))


### Bug Fixes

* improve metadata filtering error handling and query plan serialization ([faa25b6](https://github.com/alex-feel/mcp-context-server/commit/faa25b6b9ba84d20768a4377e0736ad19b7a8f86))
* remove REGEX operator and fix case sensitivity for string operators ([b6d3534](https://github.com/alex-feel/mcp-context-server/commit/b6d3534d64ec467a498cb4f7fa3c588462d950fe))

## 0.1.0 (2025-09-25)


### ⚠ BREAKING CHANGES

* add initial version

### Features

* add initial version ([ac17f19](https://github.com/alex-feel/mcp-context-server/commit/ac17f19b3cc0d6d23aaf6820c73abe588ac75da4))
